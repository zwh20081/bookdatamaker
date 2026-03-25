"""Parallel dataset generation using multiple LLM threads."""

import asyncio
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional
import threading

from openai import OpenAI
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel

from bookdatamaker.dataset.dataset_manager import DatasetManager, DuplicateEntryError, DEFAULT_DUPLICATE_THRESHOLD
from bookdatamaker.llm.message_utils import (
    extract_think,
    safe_prune_messages,
    sanitize_tool_pairs,
    serialize_messages,
)
from bookdatamaker.llm.prompt_builder import create_system_prompt as build_system_prompt
from bookdatamaker.tools.image_tools import get_image_data_url, list_page_images
from bookdatamaker.tools.page_tools import PAGE_TOOL_NAMES, execute_page_tool
from bookdatamaker.tools.registry import build_openai_tool_defs
from bookdatamaker.tools.submission_tools import (
    build_page_access_summary,
    compute_remaining_submissions,
    validate_dataset_messages,
)
from bookdatamaker.utils.page_manager import PageManager

WARNING_SIMILARITY_THRESHOLD = 50.0

console = Console()


class ParallelDatasetGenerator:
    """Generate datasets in parallel using multiple LLM threads."""

    def __init__(
        self,
        page_manager: PageManager,
        db_path: Path,
        mode: str,
        distribution: str,
        datasets_per_thread: int,
        openai_api_key: Optional[str],
        openai_api_url: str,
        model: Optional[str] = None,
        vllm_model_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        custom_prompt: Optional[str] = None,
        tool_call_parser: Optional[str] = None,
        max_messages: Optional[int] = None,
        api_delay: float = 0.0,
        extracted_dir: Optional[Path] = None,
        minimax_mcp_key: Optional[str] = None,
        search1api_key: Optional[str] = None,
    ) -> None:
        """Initialize parallel dataset generator.

        Args:
            page_manager: PageManager instance with loaded pages
            db_path: Path to SQLite database
            mode: 'api' or 'vllm'
            distribution: Distribution string (e.g., "10,10,20,30,20,10")
            datasets_per_thread: Target number of conversations per thread
            openai_api_key: OpenAI API key (for API mode)
            openai_api_url: OpenAI API URL
            model: Model name (optional, uses server default if None)
            vllm_model_path: Path to vLLM model (for vLLM mode)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum model context length (None = model's max)
            custom_prompt: Additional custom instructions for system prompt
            tool_call_parser: Tool call parser name for vLLM (required for vLLM mode)
            max_messages: Maximum message history to keep (None = unlimited)
            api_delay: Delay in seconds between API requests (default: 0.0)
            extracted_dir: Path to extracted directory for image access (optional)
            minimax_mcp_key: MiniMax API key for MCP tools (optional)
        """
        self.page_manager = page_manager
        self.db_path = db_path
        self.mode = mode
        self.distribution = parse_distribution(distribution)
        self.num_threads = len(self.distribution)  # Thread count derived from distribution
        self.datasets_per_thread = datasets_per_thread
        self.openai_api_key = openai_api_key
        self.openai_api_url = openai_api_url
        self.model = model
        self.vllm_model_path = vllm_model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.custom_prompt = custom_prompt
        self.tool_call_parser = tool_call_parser
        self.max_messages = max_messages
        self.api_delay = api_delay
        self.extracted_dir = Path(extracted_dir) if extracted_dir else None
        self.minimax_mcp_key = minimax_mcp_key
        self.minimax_mcp = None  # Initialized lazily in generate()
        self.search1api_key = search1api_key
        self.search1api_mcp = None  # Initialized lazily in generate()
        self.vllm_llm = None  # Will be initialized if using vLLM mode
        
        # Progress tracking
        self.progress_lock = None
        self.current_progress = 0
        self.pbar = None
        self.thread_pbars = {}  # Per-thread progress bars
        self.log_lock = threading.Lock()  # Lock for console output

    def _update_progress(self, increment: int = 1) -> None:
        """Update progress bar in thread-safe manner.
        
        Args:
            increment: Number of items to increment progress by
        """
        if self.pbar and self.progress_lock:
            with self.progress_lock:
                self.pbar.update(increment)
                self.current_progress += increment
    
    def _log_llm_output(self, thread_id: int, content: str, tool_calls: list = None) -> None:
        """Log LLM output in a thread-safe manner.
        
        Args:
            thread_id: Thread identifier
            content: LLM response content (may contain <think> tags)
            tool_calls: List of tool calls made by LLM
        """
        with self.log_lock:
            if content and content.strip():
                think_text, visible_text = extract_think(content)
                if think_text:
                    short_think = think_text[:150] + "..." if len(think_text) > 150 else think_text
                    tqdm.write(f"[Thread {thread_id}] 💭 Think: {short_think}")
                if visible_text:
                    short_content = visible_text[:100] + "..." if len(visible_text) > 100 else visible_text
                    tqdm.write(f"[Thread {thread_id}] LLM: {short_content}")
            
            if tool_calls:
                for tool_call in tool_calls:
                    # Handle both object (from API) and dict (from restored state)
                    if isinstance(tool_call, dict):
                        tool_name = tool_call["function"]["name"]
                    else:
                        tool_name = tool_call.function.name
                    tqdm.write(f"[Thread {thread_id}] 🔧 Tool: {tool_name}")
    
    def _log_tool_result(self, thread_id: int, tool_name: str, result: dict) -> None:
        """Log tool execution result.
        
        Args:
            thread_id: Thread identifier
            tool_name: Name of the tool
            result: Tool execution result
        """
        with self.log_lock:
            if tool_name == "submit_dataset":
                if "error" in result:
                    tqdm.write(f"[Thread {thread_id}] ❌ Submit failed: {result['error']}")
                    question = result.get("question")
                    answer = result.get("answer")
                    if question:
                        short_q = question[:60] + "..." if len(question) > 60 else question
                        tqdm.write(f"[Thread {thread_id}]    Rejected Q: {short_q}")
                    if answer:
                        short_a = answer[:60] + "..." if len(answer) > 60 else answer
                        tqdm.write(f"[Thread {thread_id}]    Rejected A: {short_a}")
                elif "warning" in result:
                    tqdm.write(f"[Thread {thread_id}] ⚠️ {result['warning']}")
                else:
                    count = result.get('count', '?')
                    remaining = result.get('remaining', 0)
                    turns = result.get('turns', 1)
                    messages = result.get('messages', [])
                    
                    # Display first two messages (first turn)
                    if len(messages) >= 2:
                        user_msg = messages[0][:60] + "..." if len(messages[0]) > 60 else messages[0]
                        assistant_msg = messages[1][:60] + "..." if len(messages[1]) > 60 else messages[1]
                        tqdm.write(f"[Thread {thread_id}] ✓ #{count} {turns}-turn conversation ({remaining} remaining)")
                        tqdm.write(f"[Thread {thread_id}]    Q: {user_msg}")
                        tqdm.write(f"[Thread {thread_id}]    A: {assistant_msg}")
                    else:
                        tqdm.write(f"[Thread {thread_id}] ✓ #{count} Submitted {turns}-turn conversation ({remaining} remaining)")
            elif tool_name == "exit":
                if result.get("rejected"):
                    tqdm.write(f"[Thread {thread_id}] ❌ Exit rejected: {result.get('remaining', 0)} pairs remaining")
                elif result.get("success"):
                    tqdm.write(f"[Thread {thread_id}] 🏁 Exit accepted: Task completed!")
                else:
                    tqdm.write(f"[Thread {thread_id}] 🏁 Exit called")
            else:
                tqdm.write(f"[Thread {thread_id}] → {tool_name} executed")

    def calculate_positions(self, total_pages: int) -> List[int]:
        """Calculate starting positions based on distribution.

        Args:
            total_pages: Total number of pages in document

        Returns:
            List of starting page numbers
        """
        # Normalize distribution to sum to 100
        total = sum(self.distribution)
        normalized = [d / total for d in self.distribution]
        
        positions = []
        cumulative = 0.0
        
        for ratio in normalized:
            position = int(cumulative * total_pages)
            positions.append(max(1, position))  # Ensure at least page 1
            cumulative += ratio
        
        return positions

    def create_system_prompt(self, start_page: int, thread_id: int) -> str:
        """Create system prompt for LLM thread.

        Args:
            start_page: Starting page number
            thread_id: Thread identifier

        Returns:
            System prompt text
        """
        return build_system_prompt(
            start_page=start_page,
            thread_id=thread_id,
            total_pages=self.page_manager.get_total_pages(),
            datasets_per_thread=self.datasets_per_thread,
            extracted_dir=self.extracted_dir,
            has_minimax_mcp=bool(self.minimax_mcp),
            has_search1api_mcp=bool(self.search1api_mcp),
            custom_prompt=self.custom_prompt,
        )
    
    def _get_mcp_tools(self) -> list:
        """Get MCP tool definitions for OpenAI function calling.
        
        Returns:
            List of tool definitions
        """
        minimax_extra = self.minimax_mcp.get_openai_tool_defs() if self.minimax_mcp else None
        search1api_extra = (
            self.search1api_mcp.get_openai_tool_defs() if self.search1api_mcp else None
        )
        return build_openai_tool_defs(
            include_image_tools=bool(self.extracted_dir),
            include_get_image=bool(self.extracted_dir and not self.minimax_mcp),
            minimax_extra=minimax_extra,
            search1api_extra=search1api_extra,
        )

    async def generate(self) -> int:
        """Run parallel dataset generation.

        Returns:
            Total number of conversations generated
        """
        from tqdm import tqdm
        import threading
        import json
        
        # Save session metadata
        with DatasetManager(str(self.db_path)) as dm:
            session_config = {
                "mode": self.mode,
                "distribution": ",".join(map(str, self.distribution)),
                "datasets_per_thread": self.datasets_per_thread,
                "num_threads": self.num_threads,
            }
            dm.set_session_metadata("config", json.dumps(session_config))
            dm.set_session_metadata("openai_api_url", self.openai_api_url)
            if self.model:
                dm.set_session_metadata("model", self.model)
            if self.vllm_model_path:
                dm.set_session_metadata("vllm_model_path", self.vllm_model_path)
        
        # Initialize MiniMax MCP proxy if key provided
        # Run in a separate thread to avoid event loop conflicts
        # (generate() is async, but MiniMaxMCPProxy uses its own sync event loops)
        if self.minimax_mcp_key:
            import concurrent.futures
            def _init_minimax():
                from bookdatamaker.llm.minimax_mcp import MiniMaxMCPProxy
                proxy = MiniMaxMCPProxy(api_key=self.minimax_mcp_key)
                proxy.get_openai_tool_defs()
                return proxy
            try:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = executor.submit(_init_minimax)
                self.minimax_mcp = future.result(timeout=30)
                # Don't shutdown executor — MCP event loops live in its thread pool
            except Exception as e:
                print(f"⚠️  Failed to initialize MiniMax MCP: {type(e).__name__}: {e}")
                self.minimax_mcp = None
        
        # Initialize Search1API MCP proxy if key provided
        if self.search1api_key:
            import concurrent.futures
            def _init_search1api():
                from bookdatamaker.llm.search1api_mcp import Search1APIMCPProxy
                proxy = Search1APIMCPProxy(api_key=self.search1api_key)
                proxy.get_openai_tool_defs()
                return proxy
            try:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = executor.submit(_init_search1api)
                self.search1api_mcp = future.result(timeout=30)
            except Exception as e:
                print(f"⚠️  Failed to initialize Search1API MCP: {type(e).__name__}: {e}")
                self.search1api_mcp = None
        
        # Initialize vLLM if needed (shared across threads)
        if self.mode == "vllm":
            print(f"Initializing vLLM with model: {self.vllm_model_path}")
            print(f"Tool call parser: {self.tool_call_parser}")
            try:
                from vllm import LLM, SamplingParams
                # vLLM instance is thread-safe and can handle parallel requests
                vllm_kwargs = {
                    "model": self.vllm_model_path,
                    "tensor_parallel_size": self.tensor_parallel_size,
                    "enable_auto_tool_choice": True,  # Always enabled for tool calling
                    "tool_call_parser": self.tool_call_parser,
                }
                if self.max_model_len is not None:
                    vllm_kwargs["max_model_len"] = self.max_model_len
                    print(f"Using custom max_model_len: {self.max_model_len}")
                
                self.vllm_llm = LLM(**vllm_kwargs)
                self.sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=2048,
                )
                print(f"✓ vLLM initialized with {self.tensor_parallel_size} GPU(s)")
                print(f"✓ Auto tool choice enabled with parser: {self.tool_call_parser}")
            except ImportError:
                print("Error: vLLM not installed. Install with: pip install vllm")
                raise
        
        # Get total pages from page manager
        total_pages = self.page_manager.get_total_pages()
        
        # Calculate starting positions
        positions = self.calculate_positions(total_pages)
        
        # Display initialization info
        console.print("\n" + "="*60)
        console.print(f"📚 Document: {total_pages} pages")
        console.print(f"🧵 Threads: {self.num_threads}")
        console.print(f"🎯 Target: {self.datasets_per_thread} Q&A pairs per thread ({self.num_threads * self.datasets_per_thread} total)")
        model_display = self.model or "server default" if self.mode == 'api' else self.vllm_model_path
        console.print(f"🤖 Model: {model_display}")
        console.print("="*60 + "\n")
        
        console.print("[bold cyan]Thread Distribution:[/bold cyan]")
        for i, pos in enumerate(positions):
            percent = (pos / total_pages) * 100
            console.print(f"  Thread {i}: Start at page [yellow]{pos}[/yellow] ([green]{percent:.1f}%[/green])")
        print()
        
        # Setup progress tracking
        total_target = self.num_threads * self.datasets_per_thread
        self.progress_lock = threading.Lock()
        self.current_progress = 0
        
        # Create progress bar
        self.pbar = tqdm(
            total=total_target,
            desc="📊 Total Q&A pairs",
            unit=" pair",
            position=0,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        # Run threads in parallel
        # Note: vLLM's generate() is synchronous, so we use ThreadPoolExecutor
        # Each thread can make requests independently, and vLLM handles batching internally
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [
                executor.submit(
                    self._run_thread,
                    thread_id=i,
                    start_page=pos,
                )
                for i, pos in enumerate(positions)
            ]
            
            # Wait for all threads to complete
            results = [future.result() for future in futures]
        
        # Close progress bar
        self.pbar.close()
        
        # Clean up MiniMax MCP proxy
        if self.minimax_mcp:
            self.minimax_mcp.close()
        
        # Clean up Search1API MCP proxy
        if self.search1api_mcp:
            self.search1api_mcp.close()
        
        # Aggregate results
        total_generated = sum(r["submitted"] for r in results)
        
        # Display final results
        print("\n" + "="*60)
        console.print("[bold green]📊 Generation Complete![/bold green]\n")
        
        for r in results:
            if r["status"] == "completed":
                status_icon = "✅"
                status_color = "green"
            else:
                status_icon = "⚠️"
                status_color = "yellow"
            
            console.print(
                f"{status_icon} Thread {r['thread_id']}: "
                f"[{status_color}]{r['submitted']}/{self.datasets_per_thread}[/{status_color}] pairs "
                f"({r['status']}) - {r.get('iterations', 0)} iterations"
            )
        
        console.print(f"\n[bold cyan]Total generated:[/bold cyan] [bold yellow]{total_generated}[/bold yellow] Q&A pairs")
        print("="*60 + "\n")
        
        return total_generated
    
    def _run_thread(self, thread_id: int, start_page: int) -> dict:
        """Run a single generation thread.

        Args:
            thread_id: Thread identifier
            start_page: Starting page number

        Returns:
            Result dictionary with statistics
        """
        try:
            system_prompt = self.create_system_prompt(start_page, thread_id)
            
            if self.mode == "api":
                # API mode: use OpenAI client with function calling
                client = OpenAI(
                    base_url=self.openai_api_url,
                    api_key=self.openai_api_key,
                    timeout=600.0,  # 600 seconds timeout for long-running requests
                )
                
                # Initialize dataset manager
                dataset_manager = DatasetManager(str(self.db_path))
                page_manager = self.page_manager  # Use the page_manager from class instance
                
                # Try to restore from checkpoint
                saved_state = dataset_manager.get_thread_state(thread_id)

                # Normalize and validate restored state for reliability
                if saved_state:
                    if not isinstance(saved_state.get("messages"), list):
                        saved_state["messages"] = []
                    try:
                        saved_state["submitted_count"] = int(saved_state.get("submitted_count", 0))
                    except (TypeError, ValueError):
                        saved_state["submitted_count"] = 0
                    try:
                        saved_state["current_position"] = int(saved_state.get("current_position", start_page))
                    except (TypeError, ValueError):
                        saved_state["current_position"] = start_page
                    if saved_state["current_position"] < 1:
                        saved_state["current_position"] = 1
                    total_pages = page_manager.get_total_pages()
                    if saved_state["current_position"] > total_pages:
                        saved_state["current_position"] = start_page
                    if saved_state["submitted_count"] < 0:
                        saved_state["submitted_count"] = 0
                    if saved_state["submitted_count"] > self.datasets_per_thread:
                        saved_state["submitted_count"] = self.datasets_per_thread
                
                # Check if thread already completed
                if saved_state and saved_state["status"] == "completed":
                    
                    # Update progress bar to reflect already completed work
                    self._update_progress(saved_state['submitted_count'])
                    
                    # Return completed state
                    return {
                        "thread_id": thread_id,
                        "submitted": saved_state["submitted_count"],
                        "status": "completed",
                        "iterations": 0  # No new iterations since already done
                    }
                
                if saved_state and saved_state["status"] != "completed":
                    # Resume from checkpoint
                    current_position = saved_state["current_position"]
                    submitted_count = saved_state["submitted_count"]
                    messages = saved_state["messages"]
                    
                    # Ensure reliable conversation bootstrap from restored state
                    if not messages:
                        messages = [{"role": "system", "content": system_prompt}]
                    elif messages[0].get("role") == "system":
                        messages[0]["content"] = system_prompt
                    else:
                        messages.insert(0, {"role": "system", "content": system_prompt})
                    
                    # Prune message history: use max_messages or default cap of 40
                    prune_limit = self.max_messages or 40
                    if len(messages) > prune_limit:
                        messages = safe_prune_messages(messages, keep_last=10)
                        with self.log_lock:
                            tqdm.write(f"[Thread {thread_id}] ✂️  Pruned restored message history to {len(messages)} messages")
                    
                    with self.log_lock:
                        tqdm.write(f"[Thread {thread_id}] 🔄 Resuming from checkpoint: {submitted_count}/{self.datasets_per_thread} pairs completed, at page {current_position}")
                    
                    # Update progress bar to reflect already completed work
                    self._update_progress(submitted_count)
                    
                    # Jump to saved position; fallback safely if invalid
                    if page_manager.jump_to_page(current_position) is None:
                        current_position = start_page
                        page_manager.jump_to_page(current_position)
                    
                    # Append resume info to system prompt instead of injecting a user message
                    remaining = self.datasets_per_thread - submitted_count
                    resume_info = f"\n\n# SESSION RESUMED\nYou have submitted {submitted_count}/{self.datasets_per_thread} conversations. You need {remaining} more. Continue from page {current_position}."
                    if self.custom_prompt:
                        resume_info += f"\nREMINDER: {self.custom_prompt}"
                    if messages and messages[0].get("role") == "system":
                        messages[0]["content"] += resume_info
                    
                    # Ensure at least one user message exists after pruning
                    # (API requires user message; pruning may leave only system msg)
                    has_user = any(m.get("role") == "user" for m in messages)
                    if not has_user:
                        messages.append({
                            "role": "user",
                            "content": f"Continue the task. You have {remaining} conversations left. Start from page {current_position}."
                        })
                else:
                    # Start fresh
                    current_position = start_page
                    start_msg = f"Please start the task. First, call jump_to_page to navigate to page {start_page}, then begin generating {self.datasets_per_thread} Q&A pairs."
                    if self.custom_prompt:
                        start_msg += f"\n\nIMPORTANT: {self.custom_prompt}"
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": start_msg}
                    ]
                    submitted_count = 0
                    
                    # Save initial state
                    dataset_manager.save_thread_state(
                        thread_id=thread_id,
                        start_position=start_page,
                        current_position=current_position,
                        submitted_count=0,
                        target_count=self.datasets_per_thread,
                        status="running",
                        messages=serialize_messages(messages)
                    )

                max_iterations = self.datasets_per_thread * 20  # Safety limit
                tools = self._get_mcp_tools()
                no_tool_call_count = 0  # Track consecutive responses without tool calls
                last_warned_hash: Optional[str] = None  # Track warned submission for re-submit acceptance
                consecutive_similarity_count = 0  # Track consecutive similarity warnings/rejections
                consecutive_duplicate_count = 0  # Track consecutive hard duplicate rejections (100%)
                empty_response_count = 0  # Track consecutive empty/no-choices responses
                
                for iteration in range(max_iterations):
                    try:
                        # Prune message history if limit exceeded (default 50 if not set)
                        prune_cap = self.max_messages or 50
                        if len(messages) > prune_cap:
                            messages = safe_prune_messages(messages, keep_last=10)
                            with self.log_lock:
                                tqdm.write(f"[Thread {thread_id}] ✂️  Pruned message history to {len(messages)} messages")
                            # Extra guard: API requires user message after system message
                            non_sys = [m for m in messages if m.get("role") != "system"]
                            if non_sys and non_sys[0].get("role") != "user":
                                insert_at = 1 if (messages and messages[0].get("role") == "system") else 0
                                messages.insert(insert_at, {"role": "user", "content": "Continue the task."})
                        
                        # Log conversation length for debugging
                        if iteration % 5 == 0:
                            with self.log_lock:
                                tqdm.write(f"[Thread {thread_id}] 💬 Iteration {iteration}, conversation messages: {len(messages)}, submitted: {submitted_count}/{self.datasets_per_thread}")
                        
                        # Build request parameters
                        request_params = {
                            "messages": messages,
                            "tools": tools,
                            "tool_choice": "auto",
                        }
                        if self.model:
                            request_params["model"] = self.model
                        
                        response = client.chat.completions.create(**request_params)

                        # Add delay after API call if configured
                        if self.api_delay > 0:
                            time.sleep(self.api_delay)

                        # Guard against non-standard API responses (e.g. raw strings)
                        if not hasattr(response, 'choices') or not response.choices:
                            empty_response_count += 1
                            backoff = min(2 ** empty_response_count, 60)  # exponential backoff, max 60s
                            with self.log_lock:
                                tqdm.write(f"[Thread {thread_id}] ⚠️  Invalid API response (no choices), retry #{empty_response_count}, waiting {backoff}s...")
                            time.sleep(backoff)
                            continue

                        empty_response_count = 0  # Reset on successful response

                        message = response.choices[0].message
                        
                        # Log LLM output
                        self._log_llm_output(thread_id, message.content, message.tool_calls)
                        
                        # Add assistant message to history (must include tool_calls if present)
                        assistant_msg = {
                            "role": "assistant",
                            "content": message.content or "",
                        }
                        if message.tool_calls:
                            assistant_msg["tool_calls"] = message.tool_calls
                        messages.append(assistant_msg)
                        
                        # Check if there are tool calls
                        if message.tool_calls and len(message.tool_calls) > 0:
                            no_tool_call_count = 0  # Reset counter
                            
                            # Process each tool call
                            for tool_call in message.tool_calls:
                                import json
                                
                                # Handle both object (from API) and dict (from restored state)
                                if isinstance(tool_call, dict):
                                    function_name = tool_call["function"]["name"]
                                    function_args = json.loads(tool_call["function"]["arguments"])
                                    tool_call_id = tool_call["id"]
                                else:
                                    function_name = tool_call.function.name
                                    function_args = json.loads(tool_call.function.arguments)
                                    tool_call_id = tool_call.id
                                
                                # Execute tool and get result
                                if function_name == "submit_dataset":
                                    messages_array = function_args.get("messages", [])

                                    validation_error = validate_dataset_messages(messages_array)
                                    if validation_error:
                                        tool_result = f"Error: {validation_error}"
                                        self._log_tool_result(thread_id, function_name, {"error": validation_error})
                                    else:
                                        import hashlib
                                        content_hash = hashlib.sha256(
                                            json.dumps(messages_array, ensure_ascii=False).encode()
                                        ).hexdigest()

                                        # Check for similarity warning (50%-85%) before add_entry
                                        formatted_for_check = [
                                            {"role": "user" if i % 2 == 0 else "assistant", "content": c.strip()}
                                            for i, c in enumerate(messages_array)
                                        ]
                                        similar = dataset_manager.find_similar_entry(
                                            formatted_for_check, WARNING_SIMILARITY_THRESHOLD
                                        )

                                        if (
                                            similar
                                            and similar["similarity"] < DEFAULT_DUPLICATE_THRESHOLD
                                            and content_hash != last_warned_hash
                                        ):
                                            # Warn but don't reject — show existing and ask to confirm
                                            last_warned_hash = content_hash
                                            sim_pct = similar["similarity"]
                                            existing_msgs = similar["messages"]
                                            existing_preview = "\n".join(
                                                f"  {m['role']}: {m['content'][:120]}" for m in existing_msgs[:4]
                                            )
                                            consecutive_similarity_count += 1
                                            web_search_hint = ""
                                            if (self.minimax_mcp or self.search1api_mcp) and consecutive_similarity_count >= 2:
                                                search_tool = "minimax_web_search" if self.minimax_mcp else "search1api_search"
                                                web_search_hint = (
                                                    "\n\n🔍 You have hit similar content multiple times in a row. "
                                                    f"STRONGLY RECOMMENDED: Call {search_tool} NOW to find fresh external examples, "
                                                    "real-world cases, or latest news related to the current topic, then build a NEW conversation "
                                                    "that combines document content with web-sourced information."
                                                )
                                            tool_result = (
                                                f"⚠️ Similar content detected ({sim_pct:.1f}% match with entry #{similar['id']}).\n"
                                                f"Existing conversation preview:\n{existing_preview}\n\n"
                                                "If you believe this is sufficiently different, call submit_dataset again with the same content to confirm.\n"
                                                "Otherwise, modify the conversation to be more distinct, or move to different pages for fresh content."
                                                + web_search_hint
                                            )
                                            self._log_tool_result(thread_id, function_name, {
                                                "warning": f"Similar to #{similar['id']} ({sim_pct:.1f}%)",
                                                "action": "warned, awaiting re-submit",
                                                "consecutive_similarity": consecutive_similarity_count,
                                            })
                                        else:
                                            # Either no warning needed or re-submitting after warning
                                            if content_hash == last_warned_hash:
                                                last_warned_hash = None  # Reset after acceptance
                                            try:
                                                entry_id = dataset_manager.add_entry(messages_array)
                                            except ValueError as ve:
                                                tool_result = f"Error: {str(ve)}"
                                                self._log_tool_result(thread_id, function_name, {"error": str(ve)})
                                            except DuplicateEntryError as duplicate_error:
                                                similarity_pct = duplicate_error.similarity
                                                existing_id = duplicate_error.existing_entry["id"]
                                                consecutive_similarity_count += 1
                                                consecutive_duplicate_count += 1
                                                web_search_hint = ""
                                                if (self.minimax_mcp or self.search1api_mcp) and consecutive_similarity_count >= 2:
                                                    search_tool = "minimax_web_search" if self.minimax_mcp else "search1api_search"
                                                    web_search_hint = (
                                                        " 🔍 Multiple consecutive duplicates detected — STOP submitting similar content! "
                                                        f"Call {search_tool} RIGHT NOW to find fresh external examples, real-world cases, "
                                                        "or latest news on this topic. Combine web results with document content to create "
                                                        "genuinely new conversations."
                                                    )
                                                # Include a snippet of what was duplicated so LLM knows what to avoid
                                                dup_preview = ""
                                                if duplicate_error.existing_entry.get("messages"):
                                                    dup_msgs = duplicate_error.existing_entry["messages"]
                                                    dup_preview = " Content already stored: " + " | ".join(
                                                        f"{m['role']}: {m['content'][:80]}" for m in dup_msgs[:2]
                                                    )
                                                tool_result = (
                                                    f"❌ DUPLICATE REJECTED (entry #{existing_id}, {similarity_pct:.1f}% match).{dup_preview} "
                                                    "You MUST NOT re-submit this content. "
                                                    "Call next_page or jump_to_page to navigate to a DIFFERENT section, then create a completely new conversation."
                                                    + web_search_hint
                                                )
                                                # After 3 consecutive duplicates, force a hard redirect via injected user message
                                                if consecutive_duplicate_count >= 3:
                                                    import random
                                                    # Pick a distant page to break out of the rut
                                                    total_pages = page_manager.total_pages if page_manager else 100
                                                    jump_target = random.randint(1, max(1, total_pages))
                                                    redirect_msg = (
                                                        f"🚨 SYSTEM: You have submitted the same duplicate content {consecutive_duplicate_count} times in a row. "
                                                        f"I am forcing you to jump to page {jump_target}. "
                                                        "Call jump_to_page({}) immediately, then read at least 2-3 pages before generating a NEW conversation on a COMPLETELY DIFFERENT topic."
                                                    ).format(jump_target)
                                                    messages.append({"role": "user", "content": redirect_msg})
                                                    consecutive_duplicate_count = 0
                                                    with self.log_lock:
                                                        tqdm.write(f"[Thread {thread_id}] 🚨 Force-redirecting to page {jump_target} after {consecutive_duplicate_count + 3} consecutive duplicates")
                                                self._log_tool_result(thread_id, function_name, {
                                                    "error": f"Duplicate entry #{existing_id} ({similarity_pct:.1f}%)",
                                                    "existing_id": existing_id,
                                                    "similarity": similarity_pct,
                                                    "consecutive_similarity": consecutive_similarity_count,
                                                })
                                            else:
                                                submitted_count += 1
                                                consecutive_similarity_count = 0  # Reset on successful submission
                                                consecutive_duplicate_count = 0  # Reset on successful submission
                                                self._update_progress(1)

                                                page_number = current_position or (
                                                    page_manager.get_current_page_number() if page_manager else None
                                                )
                                                submission_counts = {}
                                                if page_number is not None:
                                                    submission_counts = dataset_manager.record_page_submission(page_number)

                                                # Save checkpoint after each successful submission
                                                dataset_manager.save_thread_state(
                                                    thread_id=thread_id,
                                                    start_position=start_page,
                                                    current_position=current_position,
                                                    submitted_count=submitted_count,
                                                    target_count=self.datasets_per_thread,
                                                    status="running",
                                                    messages=serialize_messages(messages)
                                                )
                                                
                                                remaining = compute_remaining_submissions(
                                                    submitted_count,
                                                    self.datasets_per_thread,
                                                )
                                                turns = len(messages_array) // 2
                                                if remaining > 0:
                                                    tool_result = f"Success! Submitted {turns}-turn conversation {submitted_count}/{self.datasets_per_thread}. You need {remaining} more conversations. Continue exploring and generating."
                                                    if self.custom_prompt:
                                                        tool_result += f"\n\nREMINDER: {self.custom_prompt}"
                                                else:
                                                    tool_result = f"Success! Submitted {turns}-turn conversation {submitted_count}/{self.datasets_per_thread}. Target reached! Now call exit() to complete the task."

                                                # Evaluate page submission rankings to nudge exploration
                                                if submission_counts and len(submission_counts) > 1 and page_number is not None:
                                                    total_submissions = sum(submission_counts.values())
                                                    # Avoid early warnings until enough submissions exist for ranking
                                                    if total_submissions >= 5:
                                                        sorted_pages = sorted(
                                                            submission_counts.items(),
                                                            key=lambda item: (-item[1], item[0])
                                                        )
                                                        page_rank = next(
                                                            (idx for idx, (page, _) in enumerate(sorted_pages) if page == page_number),
                                                            None
                                                        )
                                                        if page_rank is not None:
                                                            top_threshold = max(1, min(5, max(1, int(len(sorted_pages) * 0.2))))
                                                            page_count = submission_counts.get(page_number, 0)
                                                            if page_rank < top_threshold and page_count > 0:
                                                                summary = dataset_manager.get_page_submission_summary(limit=5)
                                                                least_pages = summary.get("least_submitted_pages", [])
                                                                tool_result += (
                                                                    f"\n\n⚠️ Page {page_number} now has {page_count} submissions (rank {page_rank + 1}/{len(sorted_pages)})."
                                                                    f" Please spend more time on pages with fewer submissions, such as: {least_pages}."
                                                                    f"\nGlobal submission stats: {json.dumps(summary, ensure_ascii=False)}"
                                                                )

                                                self._log_tool_result(thread_id, function_name, {
                                                    "count": submitted_count, 
                                                    "remaining": remaining,
                                                    "turns": turns,
                                                    "messages": messages_array
                                                })
                                    
                                elif function_name == "exit":
                                    reason = function_args.get("reason", "Task completed")
                                    
                                    # Check if target is reached
                                    if submitted_count >= self.datasets_per_thread:
                                        # Target reached - mark as completed and exit
                                        dataset_manager.save_thread_state(
                                            thread_id=thread_id,
                                            start_position=start_page,
                                            current_position=current_position,
                                            submitted_count=submitted_count,
                                            target_count=self.datasets_per_thread,
                                            status="completed",
                                            messages=serialize_messages(messages)
                                        )
                                        self._log_tool_result(thread_id, function_name, {"reason": reason, "success": True})
                                        return {
                                            "thread_id": thread_id,
                                            "start_position": start_page,
                                            "submitted": submitted_count,
                                            "status": "completed",
                                            "iterations": iteration + 1,
                                            "exit_reason": reason
                                        }
                                    else:
                                        # Target not reached - reject exit
                                        remaining = compute_remaining_submissions(
                                            submitted_count,
                                            self.datasets_per_thread,
                                        )
                                        tool_result = f"Exit rejected! You've only submitted {submitted_count}/{self.datasets_per_thread} Q&A pairs. You need {remaining} more pairs before you can exit. Continue generating."
                                        self._log_tool_result(thread_id, function_name, {"rejected": True, "remaining": remaining})
                                    
                                elif function_name in PAGE_TOOL_NAMES:
                                    shared_result = execute_page_tool(
                                        page_manager,
                                        function_name,
                                        function_args,
                                        default_search_max_results=20,
                                    )

                                    if not shared_result.get("ok"):
                                        error_msg = shared_result.get("error", "Unknown tool execution error")
                                        tool_result = f"Error: {error_msg}"
                                        self._log_tool_result(thread_id, function_name, {"error": error_msg})
                                    else:
                                        if shared_result.get("current_position") is not None:
                                            current_position = shared_result["current_position"]

                                        if function_name == "get_current_page":
                                            page_info = shared_result["page_info"]
                                            tool_result = (
                                                f"Page {page_info['page_number']} (of {page_info['total_pages']}):\n"
                                                f"{page_info['content']}"
                                            )
                                            self._log_tool_result(
                                                thread_id,
                                                function_name,
                                                {"page": page_info["page_number"]},
                                            )

                                        elif function_name == "jump_to_page":
                                            page_info = shared_result["page_info"]
                                            tool_result = (
                                                f"Page {page_info['page_number']} (of {page_info['total_pages']}):\n"
                                                f"{shared_result['content']}"
                                            )
                                            self._log_tool_result(
                                                thread_id,
                                                function_name,
                                                {"page": page_info["page_number"]},
                                            )

                                        elif function_name in {"next_page", "previous_page"}:
                                            page_info = shared_result["page_info"]
                                            steps = shared_result.get("steps", function_args.get("steps", 1))
                                            tool_result = (
                                                f"Moved to page {page_info['page_number']} (of {page_info['total_pages']}):\n"
                                                f"{shared_result['content']}"
                                            )
                                            self._log_tool_result(
                                                thread_id,
                                                function_name,
                                                {"to": page_info["page_number"], "steps": steps},
                                            )

                                        elif function_name == "get_page_range":
                                            pages_data = shared_result["pages_data"]
                                            total = shared_result["total_pages"]
                                            parts = [
                                                f"--- Page {pn} (of {total}) ---\n{pages_data[pn]}"
                                                for pn in sorted(pages_data.keys())
                                            ]
                                            tool_result = "\n\n".join(parts)
                                            self._log_tool_result(
                                                thread_id,
                                                function_name,
                                                {"pages": sorted(pages_data.keys())},
                                            )

                                        elif function_name == "get_page_context":
                                            context = shared_result["context"]
                                            current_page = shared_result["current_position"]
                                            tool_result = (
                                                f"Page {current_page} with context:\nCurrent:\n{context['content']}\n"
                                            )
                                            if context.get("previous_pages"):
                                                tool_result += (
                                                    f"\nPrevious pages: {list(context['previous_pages'].keys())}"
                                                )
                                            if context.get("next_pages"):
                                                tool_result += f"\nNext pages: {list(context['next_pages'].keys())}"
                                            self._log_tool_result(
                                                thread_id,
                                                function_name,
                                                {
                                                    "page": current_page,
                                                    "before": shared_result.get("before", function_args.get("before", 1)),
                                                    "after": shared_result.get("after", function_args.get("after", 1)),
                                                },
                                            )

                                        elif function_name == "search_text":
                                            query = shared_result["query"]
                                            results = shared_result["results"]
                                            if results:
                                                lines = [f"Found {len(results)} match(es):"]
                                                for item in results:
                                                    lines.append(
                                                        f"  Page {item['page_number']}, line {item['line_number']}: {item['content'][:120]}"
                                                    )
                                                tool_result = "\n".join(lines)
                                            else:
                                                tool_result = f"No matches found for '{query}'"
                                            self._log_tool_result(
                                                thread_id,
                                                function_name,
                                                {"query": query, "matches": len(results)},
                                            )

                                        else:
                                            tool_result = "Error: Unsupported page tool"
                                            self._log_tool_result(
                                                thread_id,
                                                function_name,
                                                {"error": "Unsupported page tool"},
                                            )

                                elif function_name == "get_page_access_summary":
                                    limit = function_args.get("limit", 5)
                                    counts = dataset_manager.get_page_submission_counts()
                                    page_numbers = list(getattr(page_manager, "page_numbers", []))
                                    last_page = dataset_manager.get_session_metadata("last_submission_page")
                                    summary = build_page_access_summary(
                                        counts=counts,
                                        page_numbers=page_numbers,
                                        last_submission_page=last_page,
                                        limit=limit,
                                    )
                                    tool_result = json.dumps(summary, ensure_ascii=False, indent=2)
                                    self._log_tool_result(thread_id, function_name, {"total_submissions": summary.get("total_submissions", 0)})

                                elif function_name == "list_page_images" and self.extracted_dir:
                                    page_num = function_args["page_number"]
                                    result = list_page_images(self.extracted_dir, page_num)
                                    if result.get("ok"):
                                        tool_result = json.dumps(
                                            {
                                                "page_number": result["page_number"],
                                                "image_count": result["image_count"],
                                                "images": result["images"],
                                            },
                                            ensure_ascii=False,
                                        )
                                    else:
                                        tool_result = result.get("error", f"Page {page_num} not found")
                                    self._log_tool_result(thread_id, function_name, {"page": page_num})

                                elif function_name == "get_image" and self.extracted_dir and not self.minimax_mcp:
                                    page_num = function_args["page_number"]
                                    image_name = function_args["image_name"]
                                    result = get_image_data_url(self.extracted_dir, page_num, image_name)
                                    if result.get("ok"):
                                        tool_result = json.dumps(
                                            {
                                                "page_number": result["page_number"],
                                                "image_name": result["image_name"],
                                                "data_url": result["data_url"],
                                            },
                                            ensure_ascii=False,
                                        )
                                    else:
                                        tool_result = result.get("error", f"Image '{image_name}' not found in page {page_num}")
                                    self._log_tool_result(thread_id, function_name, {"page": page_num, "image": image_name})

                                elif self.minimax_mcp and self.minimax_mcp.is_minimax_tool(function_name):
                                    try:
                                        tool_result = self.minimax_mcp.call_tool(thread_id, function_name, function_args)
                                    except Exception as mcp_err:
                                        tool_result = f"Error calling {function_name}: {mcp_err}"
                                    self._log_tool_result(thread_id, function_name, {"result_preview": str(tool_result)[:80]})

                                elif self.search1api_mcp and self.search1api_mcp.is_search1api_tool(function_name):
                                    try:
                                        tool_result = self.search1api_mcp.call_tool(thread_id, function_name, function_args)
                                    except Exception as mcp_err:
                                        tool_result = f"Error calling {function_name}: {mcp_err}"
                                    self._log_tool_result(thread_id, function_name, {"result_preview": str(tool_result)[:80]})

                                else:
                                    tool_result = f"Error: Unknown tool {function_name}"
                                    self._log_tool_result(thread_id, function_name, {"error": tool_result})
                                
                                # Add tool result to messages - use string format for better model understanding
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": str(tool_result)
                                })
                        
                        else:
                            # No tool calls - track and remind only after prolonged inaction
                            no_tool_call_count += 1
                            
                            if no_tool_call_count >= 4:
                                # After 4 consecutive responses without tools, inject a minimal reminder
                                remaining = self.datasets_per_thread - submitted_count
                                reminder = f"You haven't used any tools. You still need to submit {remaining} more conversations. Please use get_current_page, next_page, get_page_range, or jump_to_page to explore the document, then submit_dataset to save conversations. Call exit when you reach {self.datasets_per_thread} submissions."
                                
                                if self.custom_prompt:
                                    reminder += f"\n\nREMINDER: {self.custom_prompt}"
                                
                                with self.log_lock:
                                    tqdm.write(f"[Thread {thread_id}] ⚠️  Reminding LLM to use tools ({remaining} pairs remaining)")
                                
                                messages.append({
                                    "role": "user",
                                    "content": reminder
                                })
                                no_tool_call_count = 0  # Reset after reminding

                    except Exception as e:
                        error_str = str(e)
                        
                        # Rate limit (429): wait and retry without modifying messages
                        if "429" in error_str or "rate_limit" in error_str:
                            import random
                            jitter = random.uniform(1, 8)
                            retry_delay = max(self.api_delay, 10) + jitter
                            with self.log_lock:
                                tqdm.write(f"[Thread {thread_id}] ⏳ Rate limited, waiting {retry_delay:.1f}s before retry...")
                            time.sleep(retry_delay)
                            continue
                        
                        with self.log_lock:
                            tqdm.write(f"[Thread {thread_id}] ❌ Error: {error_str[:300]}")
                        
                        # First: sanitize any incomplete tool_call groups
                        # (could happen if exception occurred mid-tool-processing)
                        messages = sanitize_tool_pairs(messages)
                        
                        # If API returns 400 bad_request, likely message format issue
                        # Aggressively prune history to recover
                        if "400" in error_str or "bad_request" in error_str:
                            messages = safe_prune_messages(messages, keep_last=4)
                            with self.log_lock:
                                tqdm.write(f"[Thread {thread_id}] ✂️  Pruned to {len(messages)} messages after API error")
                        
                        # Add error message to continue conversation
                        error_msg = "Error occurred. Please continue with the task."
                        if self.custom_prompt:
                            error_msg += f"\n\nREMINDER: {self.custom_prompt}"
                        messages.append({
                            "role": "user",
                            "content": error_msg
                        })
                        continue

                # Max iterations reached
                return {
                    "thread_id": thread_id,
                    "start_position": start_page,
                    "submitted": submitted_count,
                    "status": "incomplete - max iterations",
                    "iterations": max_iterations,
                }
                
            else:  # vllm mode
                # vLLM mode: use direct LLM inference
                # Note: vLLM.generate() is synchronous and thread-safe
                # Multiple threads can call it simultaneously, vLLM handles batching
                prompts = [
                    f"{system_prompt}\n\nBegin at page {start_page}. Use navigation tools to explore and generate {self.datasets_per_thread} Q&A pairs."
                ]
                
                submitted_count = 0
                max_iterations = self.datasets_per_thread * 15
                
                for iteration in range(max_iterations):
                    try:
                        # Synchronous call - safe in thread pool
                        outputs = self.vllm_llm.generate(prompts, self.sampling_params)
                        content = outputs[0].outputs[0].text
                        
                        # Simulate submission detection (placeholder)
                        if "submit_dataset" in content.lower():
                            submitted_count += 1
                            self._update_progress(1)  # Update progress bar
                        
                        # Check for exit condition
                        if "exit" in content.lower() or submitted_count >= self.datasets_per_thread:
                            return {
                                "thread_id": thread_id,
                                "start_position": start_page,
                                "submitted": submitted_count,
                                "status": "completed",
                                "iterations": iteration + 1,
                            }
                        
                        # Append to prompt for next iteration
                        prompts[0] += f"\n\n{content}"

                    except Exception as e:
                        print(f"Thread {thread_id} error at iteration {iteration}: {e}")
                        continue

                # Max iterations reached
                return {
                    "thread_id": thread_id,
                    "start_position": start_page,
                    "submitted": submitted_count,
                    "status": "incomplete",
                    "iterations": max_iterations,
                }

        except Exception as e:
            return {
                "thread_id": thread_id,
                "start_position": start_page,
                "submitted": 0,
                "status": "error",
                "error": str(e),
                "iterations": 0,
            }



def parse_distribution(distribution_str: str) -> List[int]:
    """Parse distribution string to list of integers.

    Args:
        distribution_str: Comma-separated numbers (e.g., "10,10,10,20,30,20,20")

    Returns:
        List of integers

    Example:
        >>> parse_distribution("10,10,10,20,30,20,20")
        [10, 10, 10, 20, 30, 20, 20]
    """
    return [int(x.strip()) for x in distribution_str.split(",")]
