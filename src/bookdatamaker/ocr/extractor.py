"""DeepSeek OCR text extraction module."""

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import httpx
from PIL import Image, ImageOps


class OCRExtractor:
    """Extract text from images using DeepSeek OCR API or local transformers model."""

    PROGRESS_FILE_NAME = ".extraction_progress.json"

    # Version-specific configuration
    _VERSION_CONFIG = {
        "1": {
            "default_model": "deepseek-ai/DeepSeek-OCR",
            "image_size": 640,
            "api_ngram_size": 30,
            "use_flash_attn": False,
        },
        "2": {
            "default_model": "deepseek-ai/DeepSeek-OCR-2",
            "image_size": 768,
            "api_ngram_size": 20,
            "use_flash_attn": True,
        },
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.deepseek.com/v1",
        mode: Literal["api", "local"] = "api",
        local_model_path: Optional[str] = None,
        batch_size: int = 8,
        device: str = "cuda",
        skip_model_load: bool = False,
        api_concurrency: int = 4,
        ocr_version: Literal["1", "2"] = "2",
    ) -> None:
        """Initialize OCR extractor.

        Args:
            api_key: DeepSeek API key (required for API mode)
            api_url: DeepSeek API base URL
            mode: "api" for API calls, "local" for self-hosted transformers model
            local_model_path: Path to local model (for local mode). Overrides ocr_version default.
            batch_size: Batch size for local transformers processing
            device: Torch device for local mode (default: "cuda")
            skip_model_load: Skip loading OCR model (for plain text extraction)
            api_concurrency: Concurrent requests for API mode (default: 4)
            ocr_version: DeepSeek OCR version: "1" for OCR-1, "2" for OCR-2 (default)
        """
        self.mode = mode
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.ocr_version = ocr_version

        # Load version-specific config
        vcfg = self._VERSION_CONFIG[ocr_version]
        self.default_model_name: str = vcfg["default_model"]
        self.image_size: int = vcfg["image_size"]
        self.api_ngram_size: int = vcfg["api_ngram_size"]
        self.use_flash_attn: bool = vcfg["use_flash_attn"]

        self.local_model_path = local_model_path or self.default_model_name
        self.batch_size = batch_size
        self.device = device
        self.skip_model_load = skip_model_load
        self.api_concurrency = api_concurrency
        self.llm = None

        if mode == "api":
            if not skip_model_load:
                # vLLM servers typically don't require authentication
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                
                self.client = httpx.AsyncClient(
                    timeout=3600.0,  # Long timeout for OCR processing
                    headers=headers,
                )
            else:
                self.client = None
        else:
            # Local mode - initialize transformers model only if needed
            if not skip_model_load:
                self._init_local_model()

    async def __aenter__(self) -> "OCRExtractor":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self.mode == "api":
            if self.client:
                await self.client.aclose()

    def _init_local_model(self) -> None:
        """Initialize local transformers model."""
        import importlib.metadata
        
        # Import torch here (only when needed for local mode)
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch not found. Please install it for local mode:\n"
                "  pip install bookdatamaker[local]"
            )
        
        # Store torch reference for later use
        self.torch = torch
        
        # Check transformers version
        try:
            transformers_version = importlib.metadata.version("transformers")
            if transformers_version != "4.46.3":
                import warnings
                warnings.warn(
                    f"Transformers version {transformers_version} detected. "
                    f"This project requires transformers==4.46.3 for optimal compatibility. "
                    f"Please install the correct version:\n"
                    f"  pip install transformers==4.46.3\n"
                    f"Continuing with current version, but unexpected issues may occur.",
                    UserWarning,
                    stacklevel=2
                )
        except importlib.metadata.PackageNotFoundError:
            raise ImportError(
                "Transformers not found. Please install it:\n"
                "  pip install transformers==4.46.3"
            )

        from transformers import AutoModel, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_model_path, 
            trust_remote_code=True
        )

        model_kwargs = {
            "trust_remote_code": True,
            "use_safetensors": True,
            "torch_dtype": torch.bfloat16,
        }
        if self.use_flash_attn:
            try:
                import flash_attn  # noqa: F401
                model_kwargs["_attn_implementation"] = "flash_attention_2"
            except ImportError:
                import warnings
                warnings.warn(
                    "flash-attn not installed. OCR-2 will run without flash_attention_2. "
                    "For better performance, install it:\n"
                    "  pip install flash-attn --no-build-isolation\n"
                    "Or install bookdatamaker[local-flash]",
                    UserWarning,
                    stacklevel=2,
                )

        self.model = AutoModel.from_pretrained(
            self.local_model_path,
            device_map="auto",
            **model_kwargs,
        )
        self.model = self.model.eval()
        
    def _filter_ocr_text(self, text: str) -> str:
        """Filter OCR text to remove lines containing [[.....]] pattern and empty lines.
        
        This removes bounding box annotations that may appear in API mode output
        and removes empty lines to make the output more compact.
        Used for plain text mode where no image cropping is needed.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Filtered text with [[.....]] lines and empty lines removed
        """
        import re
        lines = text.split('\n')
        # Filter out lines containing [[...]] pattern and empty lines
        filtered_lines = [
            line for line in lines 
            if not re.search(r'\[\[.*?\]\]', line) and line.strip()
        ]
        return '\n'.join(filtered_lines)

    def _post_process_ocr_output(
        self, text: str, page_image: Image.Image, page_dir: Path
    ) -> str:
        """Post-process OCR output: crop images from ref/det annotations and clean text.

        Matches the behavior of the model's infer(save_results=True) path:
        - Always creates ``page_dir/images/`` directory
        - Parses ``<|ref|>label<|/ref|><|det|>coords<|/det|>`` patterns
        - Draws bounding boxes on page image → ``result_with_boxes.jpg``
        - For image refs: crops region(s) → ``images/N.jpg``, replaces with ``![](images/N.jpg)``
        - For non-image refs: removes the annotation entirely
        - Replaces ``\\coloneqq`` → ``:=`` and ``\\eqqcolon`` → ``=:``

        Args:
            text: Raw OCR text with ref/det annotations
            page_image: PIL Image of the full page (used for cropping)
            page_dir: Directory for this page (e.g., extracted/page_001/)

        Returns:
            Processed text with image links and clean formatting
        """
        import re
        import ast
        import random
        from PIL import ImageDraw, ImageFont

        img_w, img_h = page_image.size
        images_dir = page_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Parse all ref/det matches (same as model's re_match())
        ref_det_pattern = re.compile(
            r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)',
            re.DOTALL,
        )
        all_matches = ref_det_pattern.findall(text)

        # Separate image vs other matches (same as model's re_match())
        matches_images = []
        matches_other = []
        all_refs = []  # (label, coords_list) for bounding box drawing
        for full_match, label, coords_str in all_matches:
            try:
                cor_list = ast.literal_eval(coords_str.strip())
            except (ValueError, SyntaxError):
                cor_list = []
            all_refs.append((label.strip(), cor_list))
            if label.strip().lower() == "image":
                matches_images.append(full_match)
            else:
                matches_other.append(full_match)

        # Draw bounding boxes on copy of page image (matches model's draw_bounding_boxes)
        img_draw = page_image.copy()
        if img_draw.mode != "RGBA":
            img_draw = img_draw.convert("RGBA")
        draw = ImageDraw.Draw(img_draw)
        overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
        draw2 = ImageDraw.Draw(overlay)
        font = ImageFont.load_default()

        img_idx = 0
        for label_type, cor_list in all_refs:
            if not isinstance(cor_list, list) or not cor_list:
                continue
            color = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 255))
            color_a = color + (20,)
            # Normalize to list of [x1,y1,x2,y2] boxes
            if isinstance(cor_list[0], list) and len(cor_list[0]) == 4:
                boxes = cor_list
            elif isinstance(cor_list[0], list) and len(cor_list[0]) == 2 and len(cor_list) == 2:
                boxes = [[cor_list[0][0], cor_list[0][1], cor_list[1][0], cor_list[1][1]]]
            else:
                boxes = cor_list if all(isinstance(b, list) and len(b) == 4 for b in cor_list) else []

            for box in boxes:
                if len(box) != 4:
                    continue
                x1 = int(box[0] / 999 * img_w)
                y1 = int(box[1] / 999 * img_h)
                x2 = int(box[2] / 999 * img_w)
                y2 = int(box[3] / 999 * img_h)

                if label_type.lower() == "image":
                    try:
                        cropped = page_image.crop((x1, y1, x2, y2))
                        cropped.save(images_dir / f"{img_idx}.jpg", "JPEG", quality=95)
                    except Exception:
                        pass
                    img_idx += 1

                try:
                    width = 4 if label_type.lower() == "title" else 2
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                    draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                    text_y = max(0, y1 - 15)
                    text_bbox = draw.textbbox((0, 0), label_type, font=font)
                    tw = text_bbox[2] - text_bbox[0]
                    th = text_bbox[3] - text_bbox[1]
                    draw.rectangle([x1, text_y, x1 + tw, text_y + th], fill=(255, 255, 255, 30))
                    draw.text((x1, text_y), label_type, font=font, fill=color)
                except Exception:
                    pass

        img_draw.paste(overlay, (0, 0), overlay)
        img_draw.convert("RGB").save(page_dir / "result_with_boxes.jpg", "JPEG", quality=95)

        # Replace image ref/det with ![](images/N.jpg) (same as model)
        processed = text
        for idx, a_match_image in enumerate(matches_images):
            processed = processed.replace(a_match_image, f"![](images/{idx}.jpg)\n")

        # Remove non-image ref/det annotations and cleanup (same as model)
        for a_match_other in matches_other:
            processed = processed.replace(a_match_other, "")
        processed = processed.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")

        # Strip stop tokens that may leak from batch generate
        stop_str = "<\uff5cend\u2581of\u2581sentence\uff5c>"
        processed = processed.replace(stop_str, "")

        # Remove empty lines
        lines = processed.split('\n')
        cleaned_lines = [line for line in lines if line.strip()]
        return '\n'.join(cleaned_lines)
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 data URL.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image as data URL (data:image/jpeg;base64,...)
        """
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Save to bytes
            import io

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            img_bytes = buffer.getvalue()

        base64_str = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_str}"

    async def extract_text(self, image_path: Path) -> str:
        """Extract text from a single image.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text content

        Raises:
            httpx.HTTPError: If API request fails (API mode)
        """
        if self.mode == "api":
            return await self._extract_text_api(image_path)
        else:
            return await self._extract_text_local(image_path)

    async def _extract_text_api(self, image_path: Path) -> str:
        """Extract text using vLLM API (OpenAI-compatible format).

        Args:
            image_path: Path to image file

        Returns:
            Extracted text content
        """
        # Encode image to base64 data URL
        image_data_url = self._encode_image(image_path)

        # Use OpenAI-compatible chat completions endpoint with base64 image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    },
                    {
                        "type": "text",
                        "text": "<|grounding|>Convert the document to markdown."
                    }
                ]
            }
        ]

        response = await self.client.post(
            f"{self.api_url}/chat/completions",
            json={
                "model": self.default_model_name,
                "messages": messages,
                "temperature": 0.0,
                "extra_body": {
                    "skip_special_tokens": False,
                    "vllm_xargs": {
                        "ngram_size": self.api_ngram_size,
                        "window_size": 90,
                        "whitelist_token_ids": [128821, 128822],
                    },
                },
            },
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    async def _extract_text_local(self, image_path: Path) -> str:
        """Extract text using local transformers model.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text content
        """
        from concurrent.futures import ThreadPoolExecutor
        
        image = Image.open(image_path).convert("RGB")
        # Use grounding prompt for markdown conversion
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "

        # Run in thread pool executor (inference is synchronous)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            output_text = await loop.run_in_executor(
                executor, lambda: self._run_inference(image, prompt)
            )

        return output_text
    
    def _run_inference(self, image: Image.Image, prompt: str) -> str:
        """Run model inference (synchronous helper).
        
        Args:
            image: PIL Image
            prompt: Prompt text
            
        Returns:
            Extracted text
        """
        inputs = self.model.prepare_inputs(
            images=[image],
            prompts=[prompt],
            tokenizer=self.tokenizer
        )
        
        # Move inputs to GPU
        inputs = {k: v.cuda() if isinstance(v, self.torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        with self.torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=8192)
        
        # Decode output
        output_text = self.tokenizer.decode(
            outputs[0].cpu().tolist(), 
            skip_special_tokens=True
        )
        
        return output_text

    def _preprocess_ocr2_single(self, image: Image.Image, prompt_text: str) -> dict:
        """Preprocess a single image for OCR-2 batch inference.

        Replicates the preprocessing from the model's infer() method,
        producing tensor inputs for generate() without calling it.

        Args:
            image: PIL Image (RGB)
            prompt_text: Prompt string containing <image> token

        Returns:
            Dict with input_ids, images_seq_mask, images_crop, images_ori,
            images_spatial_crop tensors for one image.
        """
        import math
        import sys

        # Walk up MRO to find the actual modeling module (not torch wrappers)
        model_module = None
        for cls in type(self.model).__mro__:
            mod = sys.modules.get(cls.__module__)
            if mod is not None and hasattr(mod, "format_messages"):
                model_module = mod
                break
        if model_module is None:
            raise RuntimeError(
                f"Cannot find model module with format_messages. "
                f"Model type: {type(self.model)}, module: {type(self.model).__module__}"
            )
        format_messages_fn = model_module.format_messages
        text_encode_fn = model_module.text_encode
        BasicImageTransformCls = model_module.BasicImageTransform
        dynamic_preprocess_fn = model_module.dynamic_preprocess

        conversation = [
            {"role": "<|User|>", "content": prompt_text, "images": ["__placeholder__"]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        formatted_prompt = format_messages_fn(
            conversations=conversation, sft_format="plain", system_prompt=""
        )

        patch_size = 16
        downsample_ratio = 4
        base_size = 1024
        image_size = self.image_size

        image_transform = BasicImageTransformCls(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True
        )

        image_token = "<image>"
        image_token_id = 128815
        text_splits = formatted_prompt.split(image_token)

        images_list: list = []
        images_crop_list: list = []
        images_seq_mask: list = []
        images_spatial_crop: list = []
        tokenized_str: list = []

        # Text before <image>
        tokenized_sep = text_encode_fn(self.tokenizer, text_splits[0], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        # Image processing (crop_mode=True)
        images_crop_raw = []
        if image.size[0] <= 768 and image.size[1] <= 768:
            crop_ratio = [1, 1]
        else:
            images_crop_raw, crop_ratio = dynamic_preprocess_fn(image)

        global_view = ImageOps.pad(
            image,
            (base_size, base_size),
            color=tuple(int(x * 255) for x in image_transform.mean),
        )
        images_list.append(image_transform(global_view).to(self.torch.bfloat16))

        width_crop_num, height_crop_num = crop_ratio
        images_spatial_crop.append([width_crop_num, height_crop_num])

        if width_crop_num > 1 or height_crop_num > 1:
            for crop_img in images_crop_raw:
                images_crop_list.append(
                    image_transform(crop_img).to(self.torch.bfloat16)
                )

        # Image tokens
        num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
        num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)

        tokenized_image = [image_token_id] * (num_queries_base * num_queries_base)
        tokenized_image += [image_token_id]  # separator
        if width_crop_num > 1 or height_crop_num > 1:
            tokenized_image += [image_token_id] * (
                num_queries * width_crop_num * num_queries * height_crop_num
            )
        tokenized_str += tokenized_image
        images_seq_mask += [True] * len(tokenized_image)

        # Text after <image>
        tokenized_sep = text_encode_fn(
            self.tokenizer, text_splits[-1], bos=False, eos=False
        )
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        # BOS
        tokenized_str = [0] + tokenized_str
        images_seq_mask = [False] + images_seq_mask

        # Build tensors
        input_ids = self.torch.LongTensor(tokenized_str)
        images_seq_mask_t = self.torch.tensor(images_seq_mask, dtype=self.torch.bool)

        images_ori = self.torch.stack(images_list, dim=0)
        images_spatial_crop_t = self.torch.tensor(images_spatial_crop, dtype=self.torch.long)
        if images_crop_list:
            images_crop = self.torch.stack(images_crop_list, dim=0)
        else:
            images_crop = self.torch.zeros((1, 3, base_size, base_size))

        return {
            "input_ids": input_ids,
            "images_seq_mask": images_seq_mask_t,
            "images_crop": images_crop,
            "images_ori": images_ori,
            "images_spatial_crop": images_spatial_crop_t,
        }

    def _run_batch_generate(
        self, images: List[Image.Image], prompts: List[str]
    ) -> List[str]:
        """Run batch model inference on multiple images (synchronous).

        OCR-1: uses prepare_inputs/generate.
        OCR-2: preprocesses each image independently, left-pads to uniform
        length, then calls generate() once for the whole batch.

        Args:
            images: List of PIL Images (should be RGB)
            prompts: List of prompt strings (same length as images)

        Returns:
            List of extracted texts, one per image
        """
        import sys
        import os

        if hasattr(self.model, "prepare_inputs"):
            # OCR-1 path
            inputs = self.model.prepare_inputs(
                images=images,
                prompts=prompts,
                tokenizer=self.tokenizer,
            )
            inputs = {
                k: v.cuda() if isinstance(v, self.torch.Tensor) else v
                for k, v in inputs.items()
            }
            with self.torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=8192)

            return [
                self.tokenizer.decode(outputs[i].cpu().tolist(), skip_special_tokens=True)
                for i in range(len(images))
            ]

        # OCR-2 path: batch via preprocessed tensors
        preprocessed = [
            self._preprocess_ocr2_single(img, prompt)
            for img, prompt in zip(images, prompts)
        ]

        max_len = max(p["input_ids"].shape[0] for p in preprocessed)

        batch_input_ids = []
        batch_seq_mask = []
        batch_attn_mask = []
        batch_images = []
        batch_spatial_crops = []

        for p in preprocessed:
            seq_len = p["input_ids"].shape[0]
            pad_len = max_len - seq_len

            # Left-pad for autoregressive generation
            padded_ids = self.torch.cat([
                self.torch.zeros(pad_len, dtype=self.torch.long),
                p["input_ids"],
            ])
            batch_input_ids.append(padded_ids)

            padded_mask = self.torch.cat([
                self.torch.zeros(pad_len, dtype=self.torch.bool),
                p["images_seq_mask"],
            ])
            batch_seq_mask.append(padded_mask)

            attn = self.torch.cat([
                self.torch.zeros(pad_len, dtype=self.torch.long),
                self.torch.ones(seq_len, dtype=self.torch.long),
            ])
            batch_attn_mask.append(attn)

            batch_images.append(
                (p["images_crop"].cuda(), p["images_ori"].cuda())
            )
            batch_spatial_crops.append(p["images_spatial_crop"])

        input_ids = self.torch.stack(batch_input_ids).cuda()
        images_seq_mask = self.torch.stack(batch_seq_mask).cuda()
        attention_mask = self.torch.stack(batch_attn_mask).cuda()

        # Suppress model stdout (Rich proxy) and transformers warnings
        import warnings
        import logging
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        prev_log_level = logging.root.level
        logging.root.setLevel(logging.CRITICAL)
        # Also silence transformers logger specifically
        tf_logger = logging.getLogger("transformers")
        prev_tf_level = tf_logger.level
        tf_logger.setLevel(logging.CRITICAL)
        prev_filters = warnings.filters[:]
        warnings.filterwarnings("ignore")

        try:
            with self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                with self.torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids,
                        images=batch_images,
                        images_seq_mask=images_seq_mask,
                        images_spatial_crop=self.torch.cat(batch_spatial_crops, dim=0),
                        attention_mask=attention_mask,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        max_new_tokens=8192,
                        no_repeat_ngram_size=20,
                        use_cache=True,
                    )
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            warnings.filters[:] = prev_filters
            tf_logger.setLevel(prev_tf_level)
            logging.root.setLevel(prev_log_level)

        stop_str = "<\uff5cend\u2581of\u2581sentence\uff5c>"
        prompt_len = input_ids.shape[1]
        results: List[str] = []
        for i in range(len(images)):
            text = self.tokenizer.decode(
                output_ids[i, prompt_len:].cpu().tolist(),
                skip_special_tokens=False,
            )
            # Strip all trailing stop tokens (model may generate many)
            while text.endswith(stop_str):
                text = text[: -len(stop_str)]
            results.append(text.strip())

        return results

    @classmethod
    def get_progress_file_path(cls, output_dir: Path) -> Path:
        """Get the path to the extraction progress file."""
        return output_dir / cls.PROGRESS_FILE_NAME

    @classmethod
    def load_progress(cls, output_dir: Path) -> Optional[Dict[str, Any]]:
        """Load extraction progress metadata from disk."""
        progress_file = cls.get_progress_file_path(output_dir)
        if not progress_file.exists():
            return None

        return json.loads(progress_file.read_text(encoding="utf-8"))

    @classmethod
    def list_completed_pages(cls, output_dir: Path) -> List[int]:
        """List page numbers that already have persisted OCR results."""
        completed_pages: List[int] = []

        for page_dir in sorted(output_dir.glob("page_*")):
            if not page_dir.is_dir():
                continue

            result_file = page_dir / "result.mmd"
            if not result_file.exists():
                continue

            try:
                page_num = int(page_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            completed_pages.append(page_num)

        return sorted(completed_pages)

    def _build_progress_payload(
        self,
        document_path: Path,
        prefer_text: bool,
        total_pages: int,
        completed_pages: List[int],
        failed_pages: List[int],
        status: str,
        last_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the persisted extraction progress payload."""
        return {
            "source_path": str(document_path.resolve()),
            "mode": self.mode,
            "ocr_version": self.ocr_version,
            "local_model_path": self.local_model_path,
            "device": self.device,
            "batch_size": self.batch_size,
            "prefer_text": prefer_text,
            "total_pages": total_pages,
            "completed_pages": sorted(set(completed_pages)),
            "failed_pages": sorted(set(failed_pages)),
            "status": status,
            "last_error": last_error,
        }

    def _save_progress(
        self,
        output_dir: Path,
        document_path: Path,
        prefer_text: bool,
        total_pages: int,
        completed_pages: List[int],
        failed_pages: List[int],
        status: str,
        last_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist extraction progress metadata."""
        payload = self._build_progress_payload(
            document_path=document_path,
            prefer_text=prefer_text,
            total_pages=total_pages,
            completed_pages=completed_pages,
            failed_pages=failed_pages,
            status=status,
            last_error=last_error,
        )
        progress_file = self.get_progress_file_path(output_dir)
        progress_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def _validate_existing_progress(
        self,
        output_dir: Path,
        document_path: Path,
        prefer_text: bool,
    ) -> Optional[Dict[str, Any]]:
        """Validate whether an existing progress file is compatible with the current run."""
        existing = self.load_progress(output_dir)
        if not existing:
            return None

        expected = {
            "source_path": str(document_path.resolve()),
            "mode": self.mode,
            "ocr_version": self.ocr_version,
            "local_model_path": self.local_model_path,
            "device": self.device,
            "prefer_text": prefer_text,
        }
        mismatches = []
        for key, expected_value in expected.items():
            if existing.get(key) != expected_value:
                mismatches.append(key)

        if mismatches:
            mismatch_list = ", ".join(mismatches)
            raise ValueError(
                "Existing extraction progress is incompatible with the current run: "
                f"{mismatch_list}. Clear the output directory before rerunning."
            )

        return existing

    def _load_document_results_from_output_dir(self, output_dir: Path) -> List[tuple[int, str]]:
        """Load persisted page results from the output directory."""
        results: List[tuple[int, str]] = []

        for page_num in self.list_completed_pages(output_dir):
            result_file = output_dir / f"page_{page_num:03d}" / "result.mmd"
            results.append((page_num, result_file.read_text(encoding="utf-8")))

        return results

    def _save_plain_text_page(
        self,
        output_dir: Path,
        page_num: int,
        text: str,
        image: Optional[Image.Image] = None,
        embedded_images: Optional[List[Image.Image]] = None,
    ) -> str:
        """Save a non-OCR page result using the same directory layout as OCR output."""
        page_dir = output_dir / f"page_{page_num:03d}"
        page_dir.mkdir(parents=True, exist_ok=True)

        result_file = page_dir / "result.mmd"
        filtered_text = self._filter_ocr_text(text) if self.mode == "api" else text
        result_file.write_text(filtered_text, encoding="utf-8")

        if image is not None:
            image_file = page_dir / f"page_{page_num:03d}.png"
            image.save(image_file)

        if embedded_images:
            images_dir = page_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            for idx, emb_img in enumerate(embedded_images):
                if emb_img.mode == "RGBA":
                    emb_img = emb_img.convert("RGB")
                emb_img.save(images_dir / f"{idx}.jpg", "JPEG", quality=95)

        return filtered_text

    async def extract_from_directory(
        self, directory: Path, extensions: Optional[List[str]] = None
    ) -> List[tuple[Path, str]]:
        """Extract text from all images in a directory.

        Args:
            directory: Directory containing images
            extensions: List of file extensions to process (default: common image formats)

        Returns:
            List of tuples (image_path, extracted_text)
        """
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

        image_files = []
        for ext in extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))

        # Use batch processing for local mode
        if self.mode == "local":
            return await self._extract_batch_local(sorted(image_files))

        # Concurrent processing for API mode
        return await self._extract_batch_api(sorted(image_files))

    async def _extract_batch_api(self, image_paths: List[Path]) -> List[tuple[Path, str]]:
        """Batch extract text using API with concurrency control.

        Args:
            image_paths: List of image paths

        Returns:
            List of tuples (image_path, extracted_text)
        """
        from tqdm import tqdm
        
        if not image_paths:
            return []
        
        results = []
        semaphore = asyncio.Semaphore(self.api_concurrency)
        
        async def process_with_semaphore(image_path: Path) -> tuple[Path, str]:
            async with semaphore:
                try:
                    text = await self.extract_text(image_path)
                    return (image_path, text)
                except Exception as e:
                    print(f"\nError processing {image_path}: {e}")
                    return (image_path, "")
        
        # Create tasks for all images
        tasks = [process_with_semaphore(image_path) for image_path in image_paths]
        
        # Process with progress bar
        with tqdm(total=len(tasks), desc="OCR Processing", unit="image") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)
        
        # Sort results to match input order
        path_to_result = {path: text for path, text in results}
        return [(path, path_to_result[path]) for path in image_paths]

    async def _extract_batch_local(self, image_paths: List[Path]) -> List[tuple[Path, str]]:
        """Batch extract text using local transformers model.

        Args:
            image_paths: List of image paths

        Returns:
            List of tuples (image_path, extracted_text)
        """
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        
        if not image_paths:
            return []

        all_results = []
        total_images = len(image_paths)
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        
        # Process in batches with progress bar
        with tqdm(total=total_images, desc="OCR Processing", unit="image") as pbar:
            for i in range(0, total_images, self.batch_size):
                batch_paths = image_paths[i:i + self.batch_size]
                
                # Prepare batch input
                batch_images = []
                batch_prompts = []
                valid_paths = []

                for image_path in batch_paths:
                    try:
                        image = Image.open(image_path).convert("RGB")
                        batch_images.append(image)
                        batch_prompts.append(prompt)
                        valid_paths.append(image_path)
                    except Exception as e:
                        print(f"\nError loading {image_path}: {e}")

                if not batch_images:
                    pbar.update(len(batch_paths))
                    continue

                # Run batch inference in thread pool (inference is synchronous)
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    output_texts = await loop.run_in_executor(
                        executor, lambda: self._run_batch_generate(batch_images, batch_prompts)
                    )

                # Collect batch results
                for j, text in enumerate(output_texts):
                    if j < len(valid_paths):
                        all_results.append((valid_paths[j], text))
                
                # Update progress bar
                pbar.update(len(batch_paths))

        return all_results
    
    def _run_single_inference(self, image: Image.Image, prompt: str, page_num: int = None, output_dir: Path = None) -> str:
        """Run model inference on single image (synchronous helper).
        
        Args:
            image: PIL Image
            prompt: Prompt text
            page_num: Page number (for filename)
            output_dir: Directory to save result (if None, use temp dir)
            
        Returns:
            Extracted text
        """
        import tempfile
        import os
        import sys
        from tqdm import tqdm
        
        # Determine output directory
        if output_dir is None:
            use_temp = True
            tmpdir_obj = tempfile.TemporaryDirectory()
            base_dir = Path(tmpdir_obj.name)
        else:
            use_temp = False
            base_dir = output_dir
        
        try:
            # Create subdirectory for this page
            img_name = f"page_{page_num:03d}" if page_num else "temp"
            page_dir = base_dir / img_name
            page_dir.mkdir(parents=True, exist_ok=True)
            
            # Save image in the page subdirectory
            img_path = page_dir / f"{img_name}.png"
            image.save(img_path)
            
            # Suppress model stdout (Rich proxy) and transformers warnings
            import warnings
            import logging
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
            prev_log_level = logging.root.level
            logging.root.setLevel(logging.CRITICAL)
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Call model.infer() - DeepSeek-OCR's API
                    # Model will save result to page_dir/result.mmd
                    infer_kwargs = {
                        "prompt": prompt,
                        "image_file": str(img_path),
                        "output_path": str(page_dir),
                        "base_size": 1024,
                        "image_size": self.image_size,
                        "crop_mode": True,
                        "save_results": True,
                    }
                    if self.ocr_version == "1":
                        infer_kwargs["test_compress"] = False
                    self.model.infer(self.tokenizer, **infer_kwargs)
            finally:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                logging.root.setLevel(prev_log_level)
            
            # Read result from output file (result.mmd)
            output_file = page_dir / "result.mmd"
            if output_file.exists():
                text = output_file.read_text(encoding="utf-8")
                return text
            else:
                tqdm.write(f"Warning: Output file not found at {output_file}")
                return ""
        finally:
            if use_temp:
                tmpdir_obj.cleanup()

    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs.

        Args:
            text: Input text

        Returns:
            List of paragraphs
        """
        # Split by double newlines or more
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # If no double newlines, split by single newlines
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        return paragraphs

    async def extract_from_document(
        self, document_path: Path, prefer_text: bool = False, output_dir: Path = None
    ) -> List[tuple[int, str]]:
        """Extract text from PDF or EPUB document.

        Args:
            document_path: Path to PDF or EPUB file
            prefer_text: Try to extract text directly before OCR
            output_dir: Directory to save OCR results directly (optional)

        Returns:
            List of tuples (page_number, extracted_text)
        """
        from .document_parser import get_document_page_count, iter_document_pages

        total_pages = get_document_page_count(document_path)
        completed_pages = self.list_completed_pages(output_dir) if output_dir else []
        failed_pages: List[int] = []
        progress = None

        if output_dir:
            progress = self._validate_existing_progress(output_dir, document_path, prefer_text)
            if progress:
                failed_pages = list(progress.get("failed_pages", []))
                progress_completed = [int(page) for page in progress.get("completed_pages", [])]
                completed_pages = sorted(set(completed_pages).union(progress_completed))
            progress = self._save_progress(
                output_dir=output_dir,
                document_path=document_path,
                prefer_text=prefer_text,
                total_pages=total_pages,
                completed_pages=completed_pages,
                failed_pages=failed_pages,
                status="running",
            )
            if total_pages > 0 and len(completed_pages) >= total_pages:
                completed_payload = self._save_progress(
                    output_dir=output_dir,
                    document_path=document_path,
                    prefer_text=prefer_text,
                    total_pages=total_pages,
                    completed_pages=completed_pages,
                    failed_pages=failed_pages,
                    status="completed",
                )
                progress = completed_payload

        if progress and progress.get("status") == "completed":
            return self._load_document_results_from_output_dir(output_dir)

        completed_set = set(completed_pages)
        in_memory_results: List[tuple[int, str]] = []
        image_batch: List[tuple[int, Image.Image]] = []

        from tqdm import tqdm
        pbar = tqdm(
            total=total_pages,
            desc="OCR Processing",
            unit="page",
            dynamic_ncols=True,
            mininterval=0.5,
            initial=len(completed_set),
        )

        async def flush_local_batch() -> None:
            nonlocal completed_pages
            if not image_batch:
                return

            batch_results = await self._extract_batch_from_images(
                list(image_batch),
                output_dir,
                pbar=pbar,
            )
            for page_num, text in batch_results:
                if page_num not in completed_set:
                    completed_set.add(page_num)
                    completed_pages.append(page_num)
                if output_dir is None:
                    in_memory_results.append((page_num, text))
                if output_dir:
                    self._save_progress(
                        output_dir=output_dir,
                        document_path=document_path,
                        prefer_text=prefer_text,
                        total_pages=total_pages,
                        completed_pages=completed_pages,
                        failed_pages=failed_pages,
                        status="running",
                    )
            image_batch.clear()

        try:
            for page_num, content in iter_document_pages(document_path, prefer_text=prefer_text, start_page=1):
                if page_num in completed_set:
                    continue

                if isinstance(content, tuple) and len(content) == 3:
                    text, image, embedded_images = content
                    saved_text = text
                    if output_dir:
                        saved_text = self._save_plain_text_page(
                            output_dir, page_num, text, image=image,
                            embedded_images=embedded_images,
                        )
                    completed_set.add(page_num)
                    completed_pages.append(page_num)
                    if output_dir:
                        self._save_progress(
                            output_dir=output_dir,
                            document_path=document_path,
                            prefer_text=prefer_text,
                            total_pages=total_pages,
                            completed_pages=completed_pages,
                            failed_pages=failed_pages,
                            status="running",
                        )
                    else:
                        in_memory_results.append((page_num, saved_text))
                    if pbar is not None:
                        pbar.update(1)
                    continue

                if isinstance(content, tuple) and len(content) == 2:
                    text, image = content
                    saved_text = text
                    if output_dir:
                        saved_text = self._save_plain_text_page(output_dir, page_num, text, image=image)
                    completed_set.add(page_num)
                    completed_pages.append(page_num)
                    if output_dir:
                        self._save_progress(
                            output_dir=output_dir,
                            document_path=document_path,
                            prefer_text=prefer_text,
                            total_pages=total_pages,
                            completed_pages=completed_pages,
                            failed_pages=failed_pages,
                            status="running",
                        )
                    else:
                        in_memory_results.append((page_num, saved_text))
                    if pbar is not None:
                        pbar.update(1)
                    continue

                if isinstance(content, str):
                    saved_text = content
                    if output_dir:
                        saved_text = self._save_plain_text_page(output_dir, page_num, content)
                    completed_set.add(page_num)
                    completed_pages.append(page_num)
                    if output_dir:
                        self._save_progress(
                            output_dir=output_dir,
                            document_path=document_path,
                            prefer_text=prefer_text,
                            total_pages=total_pages,
                            completed_pages=completed_pages,
                            failed_pages=failed_pages,
                            status="running",
                        )
                    else:
                        in_memory_results.append((page_num, saved_text))
                    if pbar is not None:
                        pbar.update(1)
                    continue

                if self.skip_model_load:
                    raise RuntimeError(
                        "OCR model not loaded. Cannot process images in plain text mode. "
                        "Remove --plain-text flag or use OCR mode for image-based documents."
                    )

                if self.mode == "local":
                    image_batch.append((page_num, content))
                    if len(image_batch) >= self.batch_size:
                        await flush_local_batch()
                    continue

                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    content.save(tmp.name)
                    tmp_path = Path(tmp.name)

                try:
                    text = await self.extract_text(tmp_path)
                    processed_text = text
                    if output_dir:
                        page_dir = output_dir / f"page_{page_num:03d}"
                        page_dir.mkdir(parents=True, exist_ok=True)
                        image_file = page_dir / f"page_{page_num:03d}.png"
                        content.save(image_file)
                        rgb_image = content.convert("RGB") if content.mode != "RGB" else content
                        processed_text = self._post_process_ocr_output(text, rgb_image, page_dir)
                        result_file = page_dir / "result.mmd"
                        result_file.write_text(processed_text, encoding="utf-8")

                    completed_set.add(page_num)
                    completed_pages.append(page_num)
                    if output_dir:
                        self._save_progress(
                            output_dir=output_dir,
                            document_path=document_path,
                            prefer_text=prefer_text,
                            total_pages=total_pages,
                            completed_pages=completed_pages,
                            failed_pages=failed_pages,
                            status="running",
                        )
                    else:
                        in_memory_results.append((page_num, processed_text))
                    if pbar is not None:
                        pbar.update(1)
                finally:
                    tmp_path.unlink(missing_ok=True)

            await flush_local_batch()
        except Exception as exc:
            pbar.close()
            pbar = None
            if output_dir:
                self._save_progress(
                    output_dir=output_dir,
                    document_path=document_path,
                    prefer_text=prefer_text,
                    total_pages=total_pages,
                    completed_pages=completed_pages,
                    failed_pages=failed_pages,
                    status="error",
                    last_error=str(exc),
                )
            raise

        if pbar is not None:
            pbar.close()
            pbar = None

        if output_dir:
            self._save_progress(
                output_dir=output_dir,
                document_path=document_path,
                prefer_text=prefer_text,
                total_pages=total_pages,
                completed_pages=completed_pages,
                failed_pages=failed_pages,
                status="completed",
            )
            return self._load_document_results_from_output_dir(output_dir)

        in_memory_results.sort(key=lambda item: item[0])
        return in_memory_results
    
    async def _extract_batch_from_images(
        self,
        image_pages: List[tuple[int, Image.Image]],
        output_dir: Path = None,
        pbar=None,
    ) -> List[tuple[int, str]]:
        """Batch extract text from images using local transformers model.

        When batch_size > 1, images are sent to the model in a single forward pass
        via prepare_inputs/generate for true GPU batch inference.
        When batch_size == 1, model.infer() is used with crop_mode for higher quality.

        Args:
            image_pages: List of tuples (page_number, image)
            output_dir: Directory to save OCR results (optional, for direct save)
            pbar: Optional tqdm progress bar to update (caller owns lifecycle)

        Returns:
            List of tuples (page_number, extracted_text)
        """
        from concurrent.futures import ThreadPoolExecutor
        
        if not image_pages:
            return []
        
        all_results: List[tuple[int, str]] = []
        total_pages = len(image_pages)
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        
        loop = asyncio.get_event_loop()

        if self.batch_size > 1:
            # True GPU batch inference (both OCR-1 and OCR-2)
            with ThreadPoolExecutor(max_workers=1) as executor:
                for i in range(0, total_pages, self.batch_size):
                    batch = image_pages[i : i + self.batch_size]
                    batch_images = [img.convert("RGB") for _, img in batch]
                    batch_prompts = [prompt] * len(batch)

                    output_texts = await loop.run_in_executor(
                        executor,
                        lambda imgs=batch_images, prpts=batch_prompts:
                            self._run_batch_generate(imgs, prpts),
                    )

                    for j, (page_num, image) in enumerate(batch):
                        text = output_texts[j] if j < len(output_texts) else ""
                        rgb_image = image.convert("RGB")

                        if output_dir:
                            page_dir = output_dir / f"page_{page_num:03d}"
                            page_dir.mkdir(parents=True, exist_ok=True)
                            image_file = page_dir / f"page_{page_num:03d}.png"
                            rgb_image.save(image_file)
                            processed_text = self._post_process_ocr_output(
                                text, rgb_image, page_dir
                            )
                            result_file = page_dir / "result.mmd"
                            result_file.write_text(processed_text, encoding="utf-8")
                            text = processed_text

                        all_results.append((page_num, text))
                        if pbar is not None:
                            pbar.update(1)
        else:
            # Per-page mode via model.infer() with crop_mode
            with ThreadPoolExecutor(max_workers=1) as executor:
                for page_num, image in image_pages:
                    rgb_image = image.convert("RGB")
                    output_text = await loop.run_in_executor(
                        executor,
                        lambda img=rgb_image, pn=page_num, od=output_dir:
                            self._run_single_inference(img, prompt, pn, od),
                    )
                    all_results.append((page_num, output_text))
                    if pbar is not None:
                        pbar.update(1)
        
        return all_results
