"""DeepSeek OCR text extraction module."""

import asyncio
import base64
from pathlib import Path
from typing import List, Optional, Literal
import os

import httpx
from PIL import Image


class OCRExtractor:
    """Extract text from images using DeepSeek OCR API or local transformers model."""

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
    ) -> None:
        """Initialize OCR extractor.

        Args:
            api_key: DeepSeek API key (required for API mode)
            api_url: DeepSeek API base URL
            mode: "api" for API calls, "local" for self-hosted transformers model
            local_model_path: Path to local model (for local mode)
            batch_size: Batch size for local transformers processing
            device: Torch device for local mode (default: "cuda")
            skip_model_load: Skip loading OCR model (for plain text extraction)
            api_concurrency: Concurrent requests for API mode (default: 4)
        """
        self.mode = mode
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.local_model_path = local_model_path or "deepseek-ai/DeepSeek-OCR"
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
        
        # Set CUDA device
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_model_path, 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.local_model_path,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model = self.model.eval()
        
    def _filter_ocr_text(self, text: str) -> str:
        """Filter OCR text to remove lines containing [[.....]] pattern and empty lines.
        
        This removes bounding box annotations that may appear in API mode output
        and removes empty lines to make the output more compact.
        
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
                "model": "deepseek-ai/DeepSeek-OCR",
                "messages": messages,
                "temperature": 0.0,
                "extra_body": {
                    "skip_special_tokens": False,
                    "vllm_xargs": {
                        "ngram_size": 30,
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
                        executor, lambda: self._run_batch_inference(batch_images, batch_prompts)
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
        import sys
        import io
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
            
            # Suppress model output to avoid interfering with tqdm
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            
            try:
                # Call model.infer() - DeepSeek-OCR's API
                # Model will save result to page_dir/result.mmd
                self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=str(img_path),
                    output_path=str(page_dir),
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    test_compress=False,
                    save_results=True
                )
            finally:
                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
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
        from .document_parser import extract_document_pages

        pages = extract_document_pages(document_path, prefer_text=prefer_text)
        
        # Separate text pages and image pages
        text_results = []
        image_pages = []
        
        for page_num, content in pages:
            if isinstance(content, tuple) and len(content) == 2:
                # Plain text mode: (text, image) tuple
                text, image = content
                if output_dir:
                    page_dir = output_dir / f"page_{page_num:03d}"
                    page_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save text to result.mmd (filter if API mode)
                    result_file = page_dir / "result.mmd"
                    filtered_text = self._filter_ocr_text(text) if self.mode == "api" else text
                    result_file.write_text(filtered_text, encoding="utf-8")
                    
                    # Save page image
                    image_file = page_dir / f"page_{page_num:03d}.png"
                    image.save(image_file)
                
                text_results.append((page_num, text))
            elif isinstance(content, str):
                # Text only (fallback for old code paths)
                if output_dir:
                    page_dir = output_dir / f"page_{page_num:03d}"
                    page_dir.mkdir(parents=True, exist_ok=True)
                    result_file = page_dir / "result.mmd"
                    filtered_content = self._filter_ocr_text(content) if self.mode == "api" else content
                    result_file.write_text(filtered_content, encoding="utf-8")
                text_results.append((page_num, content))
            else:
                # Image - need OCR
                image_pages.append((page_num, content))
        
        # Process images
        if image_pages:
            if self.skip_model_load:
                raise RuntimeError(
                    "OCR model not loaded. Cannot process images in plain text mode. "
                    "Remove --plain-text flag or use OCR mode for image-based documents."
                )
            if self.mode == "local":
                # Batch process all images with transformers
                image_results = await self._extract_batch_from_images(image_pages, output_dir)
            else:
                # API mode - process sequentially
                image_results = []
                import tempfile
                from tqdm import tqdm
                
                # Process with progress bar
                for page_num, image in tqdm(image_pages, desc="OCR Processing", unit="page", ncols=100):
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        image.save(tmp.name)
                        tmp_path = Path(tmp.name)
                    
                    try:
                        text = await self.extract_text(tmp_path)
                        
                        # Save to output directory if provided
                        if output_dir:
                            page_dir = output_dir / f"page_{page_num:03d}"
                            page_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Save text to result.mmd (filter in API mode)
                            result_file = page_dir / "result.mmd"
                            filtered_text = self._filter_ocr_text(text)
                            result_file.write_text(filtered_text, encoding="utf-8")
                            
                            # Save page image
                            image_file = page_dir / f"page_{page_num:03d}.png"
                            image.save(image_file)
                        
                        image_results.append((page_num, text))
                    finally:
                        tmp_path.unlink()
            
            # Combine and sort results
            all_results = text_results + image_results
            all_results.sort(key=lambda x: x[0])  # Sort by page number
            return all_results
        
        return text_results
    
    async def _extract_batch_from_images(
        self, image_pages: List[tuple[int, Image.Image]], output_dir: Path = None
    ) -> List[tuple[int, str]]:
        """Batch extract text from images using local transformers model.

        Args:
            image_pages: List of tuples (page_number, image)
            output_dir: Directory to save OCR results (optional, for direct save)

        Returns:
            List of tuples (page_number, extracted_text)
        """
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        
        if not image_pages:
            return []
        
        all_results = []
        total_pages = len(image_pages)
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        
        # Process in batches with progress bar
        # ncols=80 fixed width, mininterval=0.5 update every 0.5s minimum
        with tqdm(total=total_pages, desc="OCR Processing", unit="page", 
                  ncols=100, mininterval=0.5) as pbar:
            for i in range(0, total_pages, self.batch_size):
                batch = image_pages[i:i + self.batch_size]
                
                # Process each image in batch individually with progress updates
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    for page_num, image in batch:
                        # Process single image
                        rgb_image = image.convert("RGB")
                        
                        # Run inference in thread pool
                        output_text = await loop.run_in_executor(
                            executor, 
                            lambda img=rgb_image, pn=page_num, od=output_dir: 
                                self._run_single_inference(img, prompt, pn, od)
                        )
                        
                        # Collect result
                        all_results.append((page_num, output_text))
                        
                        # Update progress bar immediately
                        pbar.update(1)
        
        return all_results
