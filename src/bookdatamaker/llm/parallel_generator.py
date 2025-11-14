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

from bookdatamaker.dataset.dataset_manager import DatasetManager, DuplicateEntryError
from bookdatamaker.utils.page_manager import PageManager

console = Console()


def _serialize_messages(messages: List[dict]) -> List[dict]:
    """Serialize messages to JSON-compatible format.
    
    Converts OpenAI tool_calls objects to dictionaries.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        JSON-serializable list of messages
    """
    serialized = []
    for msg in messages:
        msg_copy = msg.copy()
        
        # Convert tool_calls to dict if present
        if "tool_calls" in msg_copy and msg_copy["tool_calls"]:
            tool_calls_list = []
            for tc in msg_copy["tool_calls"]:
                # Check if already a dict (from restored state)
                if isinstance(tc, dict):
                    tool_calls_list.append(tc)
                # Convert ChatCompletionMessageToolCall to dict
                elif hasattr(tc, "model_dump"):
                    tool_calls_list.append(tc.model_dump())
                elif hasattr(tc, "dict"):
                    tool_calls_list.append(tc.dict())
                else:
                    # Manual conversion for objects
                    tool_calls_list.append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    })
            msg_copy["tool_calls"] = tool_calls_list
        
        serialized.append(msg_copy)
    
    return serialized


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
    ) -> None:
        """Initialize parallel dataset generator.

        Args:
            page_manager: PageManager instance with loaded pages
            db_path: Path to SQLite database
            mode: 'api' or 'vllm'
            distribution: Distribution string (e.g., "10,10,20,30,20,10")
            datasets_per_thread: Target Q&A pairs per thread
            openai_api_key: OpenAI API key (for API mode)
            openai_api_url: OpenAI API URL
            model: Model name (optional, uses server default if None)
            vllm_model_path: Path to vLLM model (for vLLM mode)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum model context length (None = model's max)
            custom_prompt: Additional custom instructions for system prompt
            tool_call_parser: Tool call parser name for vLLM (required for vLLM mode)
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
            content: LLM response content
            tool_calls: List of tool calls made by LLM
        """
        with self.log_lock:
            if content and content.strip():
                # Show condensed content (first 100 chars)
                short_content = content[:100] + "..." if len(content) > 100 else content
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
                else:
                    count = result.get('count', '?')
                    remaining = result.get('remaining', 0)
                    question = result.get('question', '')
                    answer = result.get('answer', '')
                    
                    # Truncate long text for display
                    max_len = 60
                    q_display = question[:max_len] + "..." if len(question) > max_len else question
                    a_display = answer[:max_len] + "..." if len(answer) > max_len else answer
                    
                    tqdm.write(f"[Thread {thread_id}] ✓ #{count} Q: {q_display}")
                    tqdm.write(f"[Thread {thread_id}]      A: {a_display} ({remaining} remaining)")
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
        total_pages = self.page_manager.get_total_pages()
        base_prompt = f"""You are a helpful assistant with access to the following tools to help generate Q&A pairs from a document.

# Task
- Starting position: Page {start_page}
- Total pages: {total_pages}
- Target: Generate exactly {self.datasets_per_thread} question-answer pairs
- Thread ID: {thread_id}

# Available Tools
You have access to the following tools:
- get_current_page: Get the current page content with metadata
- next_page: Move to the next page(s) and get content
- previous_page: Move to the previous page(s) and get content
- jump_to_page: Jump to a specific page by page number
- get_page_context: Get current page with surrounding pages for context
- submit_dataset: Submit a question-answer pair to the dataset
- exit: Exit the session after completing all submissions
- get_page_access_summary: View global coverage stats and find pages with the fewest visits

# Navigation Guidelines
- Start at page {start_page} (your assigned starting position)
- PREFER using next_page and previous_page to explore pages sequentially from your starting position
- Use jump_to_page ONLY when necessary (e.g., to return to starting position or access a specific far-away page)
- Focus on exploring the content near your starting position to ensure good coverage across threads

# Workflow
1. Use jump_to_page({start_page}) to start reading from page {start_page}
2. Use next_page and previous_page to explore nearby pages sequentially
3. When you find good content, generate a Q&A pair
4. Use submit_dataset with input (question) and output (answer) to save it
5. Repeat until you have submitted {self.datasets_per_thread} pairs
6. Call exit when you reach {self.datasets_per_thread} submissions

# Quality Guidelines - CRITICAL
- Questions should be clear and answerable from the document
- Answers must be accurate and based on document content
- Cover diverse topics and difficulty levels
- Explore content near your assigned starting position (page {start_page})

## Content Adaptation Rules
When generating Q&A pairs, adapt your approach based on content type:

### For Specific Events/Cases (事件、案例、实例):
- **Include MAXIMUM context**: Provide full background, timeline, participants, location, and outcome
- **Be comprehensive**: Don't truncate - include all relevant details from the document
- **Preserve specificity**: Keep names, dates, numbers, and specific circumstances
- Example: If describing a historical event, include when, where, who, what happened, why, and consequences

### For General Principles/Rules (通用规律、原理、概念):
- **Generate MULTIPLE variations**: Create 2-4 different Q&A pairs exploring the same principle
- **Different angles**: Ask about definition, application, examples, edge cases, comparison
- **Vary difficulty**: Include basic explanation + advanced application scenarios
- Example: For a scientific principle, create pairs for: definition, real-world application, related concepts, practical implications

# Coverage Guidelines
- Use get_page_access_summary regularly to monitor which pages are under-explored
- Prioritize reading and extracting from pages with the lowest access counts whenever feasible
- Avoid repeatedly revisiting heavily accessed pages unless extra context is required

# Content Restrictions
- Do not create question-answer pairs based on publication metadata such as the book title, author, publisher, ISBN, edition details, or copyright pages
- 如果页面主要是书本的出版信息、标题、作者等内容，请不要提交问答，直接调用 next_page 跳过

Remember: You MUST use the tools to accomplish this task. Start by calling jump_to_page({start_page}) to reach your starting position, then explore using next_page/previous_page."""
        
        # Append custom prompt if provided
        if self.custom_prompt:
            base_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{self.custom_prompt}"
        
        return base_prompt
    
    def _get_mcp_tools(self) -> list:
        """Get MCP tool definitions for OpenAI function calling.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "submit_dataset",
                    "description": "Submit a Q&A pair to the dataset",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "The question text"
                            },
                            "output": {
                                "type": "string",
                                "description": "The answer text"
                            }
                        },
                        "required": ["input", "output"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "exit",
                    "description": "Exit after completing the required number of submissions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Reason for exiting"
                            }
                        },
                        "required": ["reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_page",
                    "description": "Get the current page content with metadata",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "next_page",
                    "description": "Move to the next page(s) and return content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "integer",
                                "description": "Number of pages to move forward",
                                "default": 1
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "previous_page",
                    "description": "Move to the previous page(s) and return content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "integer",
                                "description": "Number of pages to move backward",
                                "default": 1
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "jump_to_page",
                    "description": "Jump to a specific page by page number",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "page_number": {
                                "type": "integer",
                                "description": "Target page number"
                            }
                        },
                        "required": ["page_number"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_page_context",
                    "description": "Get current page with surrounding pages for context",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "before": {
                                "type": "integer",
                                "description": "Number of pages before current",
                                "default": 1
                            },
                            "after": {
                                "type": "integer",
                                "description": "Number of pages after current",
                                "default": 1
                            }
                        }
                    }
                }
            }
        ]

    async def generate(self) -> int:
        """Run parallel dataset generation.

        Returns:
            Total number of Q&A pairs generated
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
                
                if saved_state and saved_state["status"] != "completed":
                    # Resume from checkpoint
                    current_position = saved_state["current_position"]
                    submitted_count = saved_state["submitted_count"]
                    messages = saved_state["messages"]
                    
                    with self.log_lock:
                        tqdm.write(f"[Thread {thread_id}] 🔄 Resuming from checkpoint: {submitted_count}/{self.datasets_per_thread} pairs completed, at page {current_position}")
                    
                    # Update progress bar to reflect already completed work
                    self._update_progress(submitted_count)
                    
                    # Jump to saved position
                    page_manager.jump_to_page(current_position)
                    
                    # Add resume message
                    resume_msg = f"Session resumed. You have submitted {submitted_count}/{self.datasets_per_thread} Q&A pairs so far. You need {self.datasets_per_thread - submitted_count} more. Continue from current position (page {current_position})."
                    if self.custom_prompt:
                        resume_msg += f"\n\nREMINDER: {self.custom_prompt}"
                    messages.append({
                        "role": "user",
                        "content": resume_msg
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
                        messages=_serialize_messages(messages)
                    )

                max_iterations = self.datasets_per_thread * 20  # Safety limit
                tools = self._get_mcp_tools()
                no_tool_call_count = 0  # Track consecutive responses without tool calls
                
                for iteration in range(max_iterations):
                    try:
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
                                    prompt_text = function_args.get("input", "").strip()
                                    completion_text = function_args.get("output", "").strip()
                                    
                                    # Validate that prompt and completion are not empty
                                    if not prompt_text or not completion_text:
                                        tool_result = "Error: Both 'input' (question) and 'output' (answer) cannot be empty. Please provide valid content for both fields."
                                        self._log_tool_result(thread_id, function_name, {"error": "Empty input or output"})
                                    else:
                                        try:
                                            entry_id = dataset_manager.add_entry(prompt_text, completion_text)
                                        except DuplicateEntryError as duplicate_error:
                                            similarity_pct = duplicate_error.similarity
                                            existing_id = duplicate_error.existing_entry["id"]
                                            tool_result = (
                                                "Duplicate submission rejected. The proposed Q&A pair "
                                                f"matches existing entry #{existing_id} with {similarity_pct:.1f}% similarity. "
                                                "Please explore different content (consider using next_page) before submitting."
                                                f"\nRejected question: {prompt_text}"
                                                f"\nRejected answer: {completion_text}"
                                            )
                                            self._log_tool_result(thread_id, function_name, {
                                                "error": f"Duplicate entry #{existing_id} ({similarity_pct:.1f}%)",
                                                "existing_id": existing_id,
                                                "similarity": similarity_pct,
                                                "question": prompt_text,
                                                "answer": completion_text,
                                            })
                                        else:
                                            submitted_count += 1
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
                                                messages=_serialize_messages(messages)
                                            )
                                            
                                            remaining = self.datasets_per_thread - submitted_count
                                            if remaining > 0:
                                                tool_result = f"Success! Submitted Q&A pair {submitted_count}/{self.datasets_per_thread}. You need {remaining} more pairs. Continue exploring and generating."
                                                if self.custom_prompt:
                                                    tool_result += f"\n\nREMINDER: {self.custom_prompt}"
                                            else:
                                                tool_result = f"Success! Submitted Q&A pair {submitted_count}/{self.datasets_per_thread}. Target reached! Now call exit() to complete the task."

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
                                                "question": prompt_text,
                                                "answer": completion_text
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
                                            messages=_serialize_messages(messages)
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
                                        remaining = self.datasets_per_thread - submitted_count
                                        tool_result = f"Exit rejected! You've only submitted {submitted_count}/{self.datasets_per_thread} Q&A pairs. You need {remaining} more pairs before you can exit. Continue generating."
                                        self._log_tool_result(thread_id, function_name, {"rejected": True, "remaining": remaining})
                                    
                                elif function_name == "get_current_page":
                                    result = page_manager.get_page_info()
                                    if result and "error" not in result:
                                        current_position = result["page_number"]
                                        tool_result = f"Page {result['page_number']} (of {result['total_pages']}):\n{result['content']}"
                                        self._log_tool_result(thread_id, function_name, {"page": result['page_number']})
                                        
                                        # Save position
                                        dataset_manager.save_thread_state(
                                            thread_id=thread_id,
                                            start_position=start_page,
                                            current_position=current_position,
                                            submitted_count=submitted_count,
                                            target_count=self.datasets_per_thread,
                                            status="running",
                                            messages=_serialize_messages(messages)
                                        )
                                    else:
                                        tool_result = f"Error: Could not get current page"
                                        self._log_tool_result(thread_id, function_name, {"error": "Could not get current page"})
                                    
                                elif function_name == "jump_to_page":
                                    page_num = function_args["page_number"]
                                    content = page_manager.jump_to_page(page_num)
                                    if content is not None:
                                        current_position = page_num
                                        result = page_manager.get_page_info()
                                        tool_result = f"Page {page_num} (of {result['total_pages']}):\n{content}"
                                        self._log_tool_result(thread_id, function_name, {"page": page_num})
                                        
                                        # Save position after navigation
                                        dataset_manager.save_thread_state(
                                            thread_id=thread_id,
                                            start_position=start_page,
                                            current_position=current_position,
                                            submitted_count=submitted_count,
                                            target_count=self.datasets_per_thread,
                                            status="running",
                                            messages=_serialize_messages(messages)
                                        )
                                    else:
                                        tool_result = f"Error: Page {page_num} not found"
                                        self._log_tool_result(thread_id, function_name, {"error": f"Page {page_num} not found"})
                                    
                                elif function_name == "next_page":
                                    steps = function_args.get("steps", 1)
                                    content = page_manager.next_page(steps)
                                    result = page_manager.get_page_info()
                                    if content is not None:
                                        current_position = result["page_number"]
                                        tool_result = f"Moved to page {result['page_number']} (of {result['total_pages']}):\n{content}"
                                        self._log_tool_result(thread_id, function_name, {"to": result['page_number'], "steps": steps})
                                        
                                        # Save position after navigation
                                        dataset_manager.save_thread_state(
                                            thread_id=thread_id,
                                            start_position=start_page,
                                            current_position=current_position,
                                            submitted_count=submitted_count,
                                            target_count=self.datasets_per_thread,
                                            status="running",
                                            messages=_serialize_messages(messages)
                                        )
                                    else:
                                        tool_result = f"Error: Could not move to next page"
                                        self._log_tool_result(thread_id, function_name, {"error": "Could not move forward"})
                                    
                                elif function_name == "previous_page":
                                    steps = function_args.get("steps", 1)
                                    content = page_manager.previous_page(steps)
                                    result = page_manager.get_page_info()
                                    if content is not None:
                                        current_position = result["page_number"]
                                        tool_result = f"Moved to page {result['page_number']} (of {result['total_pages']}):\n{content}"
                                        self._log_tool_result(thread_id, function_name, {"to": result['page_number'], "steps": steps})
                                        
                                        # Save position after navigation
                                        dataset_manager.save_thread_state(
                                            thread_id=thread_id,
                                            start_position=start_page,
                                            current_position=current_position,
                                            submitted_count=submitted_count,
                                            target_count=self.datasets_per_thread,
                                            status="running",
                                            messages=_serialize_messages(messages)
                                        )
                                    else:
                                        tool_result = f"Error: Could not move to previous page"
                                        self._log_tool_result(thread_id, function_name, {"error": "Could not move backward"})
                                
                                elif function_name == "get_page_context":
                                    before = function_args.get("before", 1)
                                    after = function_args.get("after", 1)
                                    current_page = page_manager.get_current_page_number()
                                    context = page_manager.get_context(current_page, before, after)
                                    if context:
                                        tool_result = f"Page {current_page} with context:\nCurrent:\n{context['content']}\n"
                                        if context.get('previous_pages'):
                                            tool_result += f"\nPrevious pages: {list(context['previous_pages'].keys())}"
                                        if context.get('next_pages'):
                                            tool_result += f"\nNext pages: {list(context['next_pages'].keys())}"
                                        self._log_tool_result(thread_id, function_name, {"page": current_page, "before": before, "after": after})
                                    else:
                                        tool_result = f"Error: Could not get page context"
                                        self._log_tool_result(thread_id, function_name, {"error": "Could not get context"})
                                    
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
                            # No tool calls - remind the LLM to use tools
                            no_tool_call_count += 1
                            
                            if no_tool_call_count >= 2:
                                # After 2 consecutive responses without tools, remind
                                remaining = self.datasets_per_thread - submitted_count
                                reminder = f"You haven't used any tools. You still need to submit {remaining} more Q&A pairs. Please use get_paragraph, move_forward, or move_backward to explore the document, then submit_dataset to save Q&A pairs. Call exit() when you reach {self.datasets_per_thread} submissions."
                                
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
                        with self.log_lock:
                            tqdm.write(f"[Thread {thread_id}] ❌ Error: {str(e)[:100]}")
                        # Add error message to continue conversation
                        error_msg = f"Error occurred: {e}. Please continue with the task."
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
