"""CLI interface for Book Data Maker."""

import asyncio
import os
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from bookdatamaker.dataset import DatasetManager
from bookdatamaker.mcp import create_mcp_server
from bookdatamaker.ocr import OCRExtractor
from bookdatamaker.utils import PageManager


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Book Data Maker - Extract text and generate datasets."""
    load_dotenv()
    
    # Show help if no command provided
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default="extracted_text",
    help="Output directory for extracted text",
)
@click.option(
    "--mode",
    type=click.Choice(["api", "local"]),
    default="api",
    help="OCR mode: 'api' for DeepSeek API, 'local' for transformers",
)
@click.option(
    "--deepseek-api-key",
    envvar="DEEPSEEK_API_KEY",
    help="API key for vLLM server (optional, most vLLM servers don't require authentication)",
)
@click.option(
    "--deepseek-api-url",
    envvar="DEEPSEEK_API_URL",
    default="http://localhost:8000/v1",
    help="vLLM API URL (default: http://localhost:8000/v1)",
)
@click.option(
    "--local-model-path",
    envvar="LOCAL_OCR_MODEL",
    default="deepseek-ai/DeepSeek-OCR",
    help="Path to local OCR model for transformers (default: deepseek-ai/DeepSeek-OCR)",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    help="Batch size for local transformers processing (default: 8)",
)
@click.option(
    "--api-concurrency",
    type=int,
    default=4,
    help="Concurrent requests for API mode (default: 4)",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Torch device for local mode (default: cuda). Examples: cuda, cuda:0, cpu",
)
@click.option(
    "--plain-text",
    is_flag=True,
    default=False,
    help="Extract plain text directly from PDF/EPUB without OCR (faster but may miss images)",
)
def extract(
    input_path: Path,
    output_dir: Path,
    mode: str,
    deepseek_api_key: Optional[str],
    deepseek_api_url: str,
    local_model_path: str,
    batch_size: int,
    api_concurrency: int,
    device: str,
    plain_text: bool,
) -> None:
    """Extract text from documents using OCR or plain text extraction.

    INPUT_PATH: Path to image file, PDF, EPUB, or directory containing files
    """
    asyncio.run(
        _extract_async(
            input_path,
            output_dir,
            mode,
            deepseek_api_key,
            deepseek_api_url,
            local_model_path,
            batch_size,
            api_concurrency,
            device,
            plain_text,
        )
    )


async def _extract_async(
    input_path: Path,
    output_dir: Path,
    mode: str,
    api_key: Optional[str],
    api_url: str,
    local_model_path: str,
    batch_size: int,
    api_concurrency: int,
    device: str,
    plain_text: bool,
) -> None:
    """Async extraction logic."""
    from bookdatamaker.utils import StatusIndicator
    
    output_dir.mkdir(parents=True, exist_ok=True)

    with StatusIndicator() as status:
        # Determine if we need OCR based on file type and flags
        is_epub = input_path.is_file() and input_path.suffix.lower() == ".epub"
        needs_ocr = not plain_text and not is_epub
        
        if plain_text:
            mode_label = "Plain Text"
        elif is_epub:
            mode_label = "EPUB (Plain Text)"
        else:
            mode_label = "local transformers" if mode == "local" else "API"
        status.print_info(f"Mode: {mode_label}")
        if mode == "local" and needs_ocr:
            status.print_info(f"Device: {device}")
        status.print_info(f"Extracting text from: {input_path}")

        # Skip OCR model loading for plain text and EPUB
        async with OCRExtractor(
            api_key=api_key,
            api_url=api_url,
            mode=mode,
            local_model_path=local_model_path,
            batch_size=batch_size,
            api_concurrency=api_concurrency,
            device=device,
            skip_model_load=not needs_ocr,
        ) as extractor:
            if input_path.is_file():
                # Check if it's a document (PDF/EPUB) or image
                if input_path.suffix.lower() in [".pdf", ".epub"]:
                    # EPUB always uses plain text extraction (no images to OCR)
                    if input_path.suffix.lower() == ".epub":
                        status.print_info(f"Processing EPUB (plain text mode)")
                        results = await extractor.extract_from_document(
                            input_path, prefer_text=True, output_dir=output_dir
                        )
                    # PDF: check plain_text flag
                    elif plain_text:
                        status.print_info(f"Processing PDF (plain text mode)")
                        results = await extractor.extract_from_document(
                            input_path, prefer_text=True, output_dir=output_dir
                        )
                    else:
                        status.print_info(f"Processing PDF (OCR mode)")
                        results = await extractor.extract_from_document(
                            input_path, prefer_text=False, output_dir=output_dir
                        )

                    if not results:
                        status.print_warning("No pages extracted")
                        return

                    status.print_success(
                        f"Extracted {len(results)} pages to: {output_dir}"
                    )

                else:
                    # Single image file
                    status.print_info("Extracting text from image...")
                    text = await extractor.extract_text(input_path)

                    output_file = output_dir / f"{input_path.stem}.txt"
                    output_file.write_text(text, encoding="utf-8")

                    status.print_success(f"Saved to: {output_file}")

            else:
                # Directory
                results = await extractor.extract_from_directory(input_path)

                if not results:
                    status.print_warning("No images found in directory")
                    return

                # Save files with progress bar
                from tqdm import tqdm
                for image_path, text in tqdm(results, desc="Saving files", unit="file"):
                    output_file = output_dir / f"{image_path.stem}.txt"
                    output_file.write_text(text, encoding="utf-8")

                status.print_success(f"Extracted {len(results)} files to: {output_dir}")

            # Save combined text - load all pages into memory with page markers
            all_files = sorted(output_dir.glob("*.txt"))

            combined_content = []

            for txt_file in all_files:
                # Extract page number from filename (e.g., page_001.txt)
                page_marker = txt_file.stem
                content = txt_file.read_text(encoding="utf-8")

                # Add page marker
                combined_content.append(f"[{page_marker.upper()}]")
                combined_content.append(content)
                combined_content.append("")  # Empty line between pages

            combined_file = output_dir / "combined.txt"
            combined_file.write_text("\n".join(combined_content), encoding="utf-8")
            status.print_success(f"Combined text with page markers saved to: {combined_file}")


@cli.command()
@click.argument("extracted_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--db",
    "-d",
    type=click.Path(path_type=Path),
    default="dataset.db",
    help="SQLite database file for dataset storage",
)
@click.option(
    "--mode",
    type=click.Choice(["api", "vllm"]),
    default="api",
    help="LLM mode: 'api' for API endpoints, 'vllm' for direct vLLM",
)
@click.option(
    "--distribution",
    default="10,10,20,30,20,10",
    help="Position distribution (comma-separated percentages, e.g., '10,10,10,20,30,20,20'). Number of values determines thread count.",
)
@click.option(
    "--datasets-per-thread",
    type=int,
    default=10,
    help="Target number of conversations per thread",
)
@click.option(
    "--openai-api-key",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key (required for API mode)",
)
@click.option(
    "--openai-api-url",
    envvar="OPENAI_API_URL",
    default="https://api.openai.com/v1",
    help="OpenAI API URL or vLLM server URL",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="LLM model name (optional, uses server default if not specified)",
)
@click.option(
    "--vllm-model-path",
    envvar="VLLM_MODEL_PATH",
    help="Path to vLLM model (required for vllm mode)",
)
@click.option(
    "--tensor-parallel-size",
    type=int,
    default=1,
    help="Number of GPUs for vLLM tensor parallelism",
)
@click.option(
    "--max-model-len",
    type=int,
    help="Maximum model context length for vLLM (default: model's max). Reduce if OOM errors occur.",
)
@click.option(
    "--custom-prompt",
    help="Additional custom instructions to append to system prompt",
)
@click.option(
    "--tool-call-parser",
    type=str,
    help="Tool call parser name for vLLM mode (required when using vllm mode). Example: 'hermes', 'mistral'",
)
@click.option(
    "--max-messages",
    type=int,
    default=None,
    help="Maximum message history to keep (only last N messages). When limit reached, keeps system prompt + last 10 messages. Helps prevent token overflow.",
)
@click.option(
    "--api-delay",
    type=float,
    default=0.0,
    help="Delay in seconds between API requests (API mode only). Use to avoid rate limits. Example: 0.5 for 500ms delay.",
)
def generate(
    extracted_dir: Path,
    db: Path,
    mode: str,
    distribution: str,
    datasets_per_thread: int,
    openai_api_key: Optional[str],
    openai_api_url: str,
    model: Optional[str],
    vllm_model_path: Optional[str],
    tensor_parallel_size: int,
    max_model_len: Optional[int],
    custom_prompt: Optional[str],
    tool_call_parser: Optional[str],
    max_messages: Optional[int],
    api_delay: float,
) -> None:
    """Generate dataset using parallel LLM threads with MCP navigation.

    EXTRACTED_DIR: Path to directory containing page_XXX/ subdirectories (from extract command)
    
    This command starts multiple LLM threads at different positions in the document.
    Each thread navigates the document using MCP tools and generates conversations.
    
    Examples:
        # API mode
        bookdatamaker generate combined.txt -d dataset.db -t 6 --datasets-per-thread 20
        
        # vLLM mode
        bookdatamaker generate combined.txt -d dataset.db --mode vllm \
            --vllm-model-path meta-llama/Llama-3-8B-Instruct -t 4
    """
    if mode == "api" and not openai_api_key:
        click.echo("Error: OpenAI API key required for API mode", err=True)
        raise click.Abort()
    
    if mode == "vllm":
        if not vllm_model_path:
            click.echo("Error: --vllm-model-path required for vllm mode", err=True)
            raise click.Abort()
        if not tool_call_parser:
            click.echo("Error: --tool-call-parser required for vllm mode", err=True)
            raise click.Abort()

    asyncio.run(
        _generate_async(
            extracted_dir,
            db,
            mode,
            distribution,
            datasets_per_thread,
            openai_api_key,
            openai_api_url,
            model,
            vllm_model_path,
            tensor_parallel_size,
            max_model_len,
            custom_prompt,
            tool_call_parser,
            max_messages,
            api_delay,
        )
    )


async def _generate_async(
    extracted_dir: Path,
    db: Path,
    mode: str,
    distribution: str,
    datasets_per_thread: int,
    openai_api_key: Optional[str],
    openai_api_url: str,
    model: str,
    vllm_model_path: Optional[str],
    tensor_parallel_size: int,
    max_model_len: Optional[int],
    custom_prompt: Optional[str],
    tool_call_parser: Optional[str],
    max_messages: Optional[int],
    api_delay: float,
) -> None:
    """Async implementation of parallel dataset generation."""
    from bookdatamaker.llm.parallel_generator import ParallelDatasetGenerator
    from bookdatamaker.dataset.dataset_manager import DatasetManager

    # Check for incomplete session
    if db.exists():
        with DatasetManager(str(db)) as dataset_manager:
            incomplete_threads = dataset_manager.get_incomplete_threads()
            if incomplete_threads:
                click.echo(f"\n⚠️  Found {len(incomplete_threads)} incomplete thread(s) in database:")
                for thread in incomplete_threads:
                    click.echo(f"  Thread {thread['thread_id']}: {thread['submitted_count']}/{thread['target_count']} pairs, last updated {thread['last_updated']}")
                
                if click.confirm("\nDo you want to resume from checkpoint?", default=True):
                    click.echo("✓ Resuming from checkpoint...")
                else:
                    if click.confirm("Clear checkpoint data and start fresh?", default=False):
                        dataset_manager.clear_thread_states()
                        click.echo("✓ Checkpoint data cleared")
                    else:
                        click.echo("Aborted.")
                        return

    click.echo(f"Loading pages from directory: {extracted_dir}")
    
    # Load pages from directory
    page_manager = PageManager.from_directory(extracted_dir)
    total_paragraphs = page_manager.total_paragraphs
    total_pages = page_manager.get_total_pages()
    
    click.echo(f"✓ Loaded {total_pages} pages, {total_paragraphs} paragraphs")

    # Initialize parallel generator
    generator = ParallelDatasetGenerator(
        page_manager=page_manager,
        db_path=db,
        mode=mode,
        distribution=distribution,
        datasets_per_thread=datasets_per_thread,
        openai_api_key=openai_api_key,
        openai_api_url=openai_api_url,
        model=model,
        vllm_model_path=vllm_model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        custom_prompt=custom_prompt,
        tool_call_parser=tool_call_parser,
        max_messages=max_messages,
        api_delay=api_delay,
    )

    click.echo(f"\nStarting {generator.num_threads} parallel threads")
    click.echo(f"Distribution: {distribution}")
    click.echo(f"Target: {datasets_per_thread} Q&A pairs per thread")

    # Run parallel generation
    try:
        total_generated = await generator.generate()
        click.echo(f"\n✓ Successfully generated {total_generated} Q&A pairs")
        click.echo(f"✓ Dataset saved to: {db}")
        click.echo(f"\nTo export, use: bookdatamaker export-dataset {db} -o output.parquet")
    except Exception as e:
        click.echo(f"\n✗ Error during generation: {e}", err=True)
        raise


@cli.command()
@click.argument("extracted_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--db",
    "-d",
    type=click.Path(path_type=Path),
    default="dataset.db",
    help="SQLite database file for dataset storage",
)
def mcp_server(extracted_dir: Path, db: Path) -> None:
    """Start MCP server with page navigation support.

    EXTRACTED_DIR: Path to directory containing page_XXX/ subdirectories (from extract command)
    
    The MCP server allows LLMs to navigate the document by pages and submit conversations
    to build a dataset. All navigation methods return unified responses with
    page_number and content.
    
    Dataset submissions are stored in SQLite database. Use 'export-dataset'
    command to export the data after the server stops.
    """
    from bookdatamaker.mcp import create_mcp_server

    click.echo(f"Loading document from: {extracted_dir}")
    click.echo(f"Dataset database: {db}")

    # Load with PageManager from directory
    try:
        page_manager = PageManager.from_directory(extracted_dir)
        stats = page_manager.get_statistics()
        
        click.echo(f"✓ Loaded {stats['total_pages']} pages")
        click.echo(f"✓ Total lines: {stats['total_lines']}")
        click.echo(f"✓ Total paragraphs: {stats['total_paragraphs']}")
        click.echo(f"✓ Total characters: {stats['total_characters']}")
        click.echo("\nStarting MCP server with page navigation support...")
        click.echo("Use Ctrl+C to stop the server\n")
        
        asyncio.run(_run_mcp_server(page_manager=page_manager, db_path=str(db)))

    except Exception as e:
        click.echo(f"✗ Failed to load document: {e}")
        raise click.Abort()


async def _run_mcp_server(
    paragraphs: Optional[list[str]] = None,
    page_manager: Optional[PageManager] = None,
    db_path: Optional[str] = None
) -> None:
    """Run MCP server."""
    server = await create_mcp_server(
        paragraphs=paragraphs, 
        page_manager=page_manager,
        db_path=db_path
    )
    await server.run()


@cli.command()
@click.argument("db_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output file path",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["jsonl", "parquet", "csv", "json"], case_sensitive=False),
    default=None,
    help="Export format (default: auto-detect from file extension)",
)
@click.option(
    "--include-metadata",
    is_flag=True,
    help="Include metadata and timestamps in export",
)
@click.option(
    "--compression",
    "-c",
    type=click.Choice(["zstd", "snappy", "gzip", "brotli", "none"], case_sensitive=False),
    default="zstd",
    help="Compression method for Parquet format (default: zstd)",
)
def export_dataset(
    db_file: Path,
    output: Path,
    format: str,
    include_metadata: bool,
    compression: str
) -> None:
    """Export dataset from SQLite database to various formats.

    DB_FILE: Path to SQLite database file (created by mcp-server)
    
    Supported formats:
    - jsonl: JSON Lines format (one JSON object per line)
    - parquet: Apache Parquet format (columnar storage)
    - csv: Comma-separated values
    - json: JSON array format
    
    Examples:
        bookdatamaker export-dataset dataset.db -o output.jsonl
        bookdatamaker export-dataset dataset.db -o output.parquet -f parquet
        bookdatamaker export-dataset dataset.db -o output.csv -f csv --include-metadata
    """
    from bookdatamaker.dataset import DatasetManager
    from bookdatamaker.utils import StatusIndicator

    with StatusIndicator() as status:
        status.print_info(f"Loading dataset from: {db_file}")
        
        try:
            with DatasetManager(str(db_file)) as dm:
                # Check entry count
                count = dm.count_entries()
                
                if count == 0:
                    status.print_warning("No entries found in database")
                    return
                
                # Auto-detect format from file extension if not specified
                if format is None:
                    ext = output.suffix.lower()
                    format_map = {
                        '.jsonl': 'jsonl',
                        '.parquet': 'parquet',
                        '.csv': 'csv',
                        '.json': 'json'
                    }
                    format = format_map.get(ext, 'jsonl')
                    status.print_info(f"Auto-detected format from extension: {format}")
                
                status.print_info(f"Found {count} entries")
                status.print_info(f"Exporting to: {output}")
                status.print_info(f"Format: {format.upper()}")
                if format.lower() == "parquet":
                    status.print_info(f"Compression: {compression}")
                
                # Export based on format
                if format.lower() == "jsonl":
                    exported = dm.export_jsonl(str(output))
                elif format.lower() == "parquet":
                    exported = dm.export_parquet(str(output), include_metadata=include_metadata, compression=compression)
                elif format.lower() == "csv":
                    exported = dm.export_csv(str(output), include_metadata=include_metadata)
                elif format.lower() == "json":
                    exported = dm.export_json(str(output), include_metadata=include_metadata)
                else:
                    status.print_error(f"Unsupported format: {format}")
                    return
                
                status.print_success(f"Exported {exported} entries to {output}")
                
                # Show file size
                file_size = output.stat().st_size
                if file_size < 1024:
                    size_str = f"{file_size} bytes"
                elif file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.2f} KB"
                else:
                    size_str = f"{file_size / (1024 * 1024):.2f} MB"
                
                status.print_info(f"File size: {size_str}")
                
        except Exception as e:
            status.print_error(f"Export failed: {e}")
            raise click.Abort()


@cli.command()
@click.argument("extracted_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--db",
    "-d",
    type=click.Path(path_type=Path),
    default="dataset.db",
    help="SQLite database file path (optional, for submit_dataset tool)",
)
@click.option(
    "--openai-api-key",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key",
)
@click.option(
    "--openai-api-url",
    envvar="OPENAI_API_URL",
    default="https://api.openai.com/v1",
    help="OpenAI API URL",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model name (optional, uses server default if not specified)",
)
def chat(
    extracted_dir: Path,
    db: Path,
    openai_api_key: Optional[str],
    openai_api_url: str,
    model: Optional[str],
) -> None:
    """Interactive chat with document using MCP tools.
    
    Chat with an LLM that can access the document through page navigation tools.
    Useful for exploring documents and testing Q&A generation interactively.
    
    Example:
        bookdatamaker chat ./extracted --model gpt-4
    """
    from bookdatamaker.utils.status import StatusIndicator
    
    status = StatusIndicator()
    
    # Validate API key
    if not openai_api_key:
        status.print_error("OpenAI API key required. Set OPENAI_API_KEY environment variable or use --openai-api-key")
        raise click.Abort()
    
    try:
        asyncio.run(_chat_async(
            extracted_dir=extracted_dir,
            db_path=db,
            openai_api_key=openai_api_key,
            openai_api_url=openai_api_url,
            model=model,
        ))
    except KeyboardInterrupt:
        status.print_info("\nChat session ended by user")
    except Exception as e:
        status.print_error(f"Chat failed: {e}")
        raise


async def _chat_async(
    extracted_dir: Path,
    db_path: Path,
    openai_api_key: str,
    openai_api_url: str,
    model: str,
) -> None:
    """Run interactive chat session with MCP tools."""
    import json
    from openai import OpenAI
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    
    console = Console()
    
    # Load document from directory
    page_manager = PageManager.from_directory(extracted_dir)
    dataset_manager = DatasetManager(str(db_path))
    
    # Initialize OpenAI client
    client = OpenAI(base_url=openai_api_url, api_key=openai_api_key)
    
    # Get MCP tools from parallel_generator
    from bookdatamaker.llm.parallel_generator import ParallelDatasetGenerator
    
    # Create a dummy generator instance to get tool definitions
    dummy_gen = ParallelDatasetGenerator(
        page_manager=page_manager,
        db_path=db_path,
        mode="api",
        distribution="100",
        datasets_per_thread=1,
        openai_api_key=openai_api_key,
        openai_api_url=openai_api_url,
        model=model,
    )
    tools = dummy_gen._get_mcp_tools()
    
    # System prompt
    total_pages = page_manager.get_total_pages()
    system_prompt = f"""You are a helpful assistant with access to document navigation tools.

Document: {extracted_dir.name}
Total pages: {total_pages}

Available tools:
- get_current_page: Get the current page content
- next_page: Move to next page(s)
- previous_page: Move to previous page(s)
- jump_to_page: Jump to a specific page number
- get_page_context: Get current page with surrounding pages
- submit_dataset: Submit a multi-turn conversation to the database (messages array: alternating user/assistant)
- exit: End the conversation

You can explore the document, answer questions, and help generate multi-turn conversations.
Use the tools to access document content when needed.

For submit_dataset, provide a "messages" array with alternating user/assistant messages, starting with user and ending with assistant.
Example: ["What is X?", "X is...", "Can you explain more?", "Sure, X also..."]"""
    
    messages = [{"role": "system", "content": system_prompt}]
    current_position = 1
    
    # Display welcome message
    console.print(Panel(
        f"[bold cyan]Interactive Document Chat[/bold cyan]\n\n"
        f"📚 Document: {extracted_dir.name}\n"
        f"📊 Pages: {total_pages}\n"
        f"🤖 Model: {model}\n"
        f"💾 Database: {db_path}\n\n"
        f"[dim]Type your questions or commands. The AI can use tools to explore the document.\n"
        f"Press Ctrl+C to exit.[/dim]",
        border_style="cyan"
    ))
    
    while True:
        try:
            # Get user input
            console.print("\n[bold green]You:[/bold green] ", end="")
            user_input = input().strip()
            
            if not user_input:
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Get LLM response
            console.print("\n[bold blue]Assistant:[/bold blue]")
            
            # Build request parameters
            request_params = {
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
            }
            if model:
                request_params["model"] = model
            
            response = client.chat.completions.create(**request_params)
            
            message = response.choices[0].message
            
            # Add assistant message to history
            assistant_msg = {
                "role": "assistant",
                "content": message.content or "",
            }
            if message.tool_calls:
                assistant_msg["tool_calls"] = message.tool_calls
            messages.append(assistant_msg)
            
            # Display assistant content
            if message.content:
                console.print(Markdown(message.content))
            
            # Process tool calls
            if message.tool_calls:
                console.print("\n[dim]Tool calls:[/dim]")
                
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    console.print(f"  🔧 {function_name}({', '.join(f'{k}={v}' for k, v in function_args.items())})")
                    
                    # Execute tool
                    if function_name == "submit_dataset":
                        messages_array = function_args.get("messages", [])
                        
                        # Validate messages format
                        if not messages_array:
                            tool_result = "Error: messages array cannot be empty"
                        elif len(messages_array) < 2:
                            tool_result = "Error: messages must contain at least one user-assistant pair (minimum 2 messages)"
                        elif len(messages_array) % 2 != 0:
                            tool_result = "Error: messages must have even length (alternating user-assistant pairs)"
                        else:
                            try:
                                entry_id = dataset_manager.add_entry(messages_array)
                                turns = len(messages_array) // 2
                                tool_result = f"Success! {turns}-turn conversation saved to database (ID: {entry_id})"
                            except ValueError as e:
                                tool_result = f"Error: {str(e)}"
                        
                    elif function_name == "exit":
                        console.print("\n[yellow]Assistant requested to exit. Goodbye![/yellow]")
                        return
                        
                    elif function_name == "get_current_page":
                        result = page_manager.get_page_info()
                        if result and "error" not in result:
                            current_position = result["page_number"]
                            tool_result = f"Page {result['page_number']} (of {result['total_pages']}):\n{result['content']}"
                        else:
                            tool_result = f"Error: Could not get current page"
                        
                    elif function_name == "jump_to_page":
                        page_num = function_args["page_number"]
                        content = page_manager.jump_to_page(page_num)
                        if content is not None:
                            current_position = page_num
                            result = page_manager.get_page_info()
                            tool_result = f"Page {page_num} (of {result['total_pages']}):\n{content}"
                        else:
                            tool_result = f"Error: Page {page_num} not found"
                        
                    elif function_name == "next_page":
                        steps = function_args.get("steps", 1)
                        content = page_manager.next_page(steps)
                        result = page_manager.get_page_info()
                        if content is not None:
                            current_position = result["page_number"]
                            tool_result = f"Moved to page {result['page_number']} (of {result['total_pages']}):\n{content}"
                        else:
                            tool_result = f"Error: Could not move to next page"
                        
                    elif function_name == "previous_page":
                        steps = function_args.get("steps", 1)
                        content = page_manager.previous_page(steps)
                        result = page_manager.get_page_info()
                        if content is not None:
                            current_position = result["page_number"]
                            tool_result = f"Moved to page {result['page_number']} (of {result['total_pages']}):\n{content}"
                        else:
                            tool_result = f"Error: Could not move to previous page"
                    
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
                        else:
                            tool_result = f"Error: Could not get page context"
                        
                    else:
                        tool_result = f"Error: Unknown tool {function_name}"
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_result)
                    })
                
                # Get follow-up response with tool results
                console.print("\n[bold blue]Assistant:[/bold blue]")
                
                # Build request parameters
                request_params = {
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                }
                if model:
                    request_params["model"] = model
                
                response = client.chat.completions.create(**request_params)
                
                message = response.choices[0].message
                
                # Add to history
                assistant_msg = {
                    "role": "assistant",
                    "content": message.content or "",
                }
                if message.tool_calls:
                    assistant_msg["tool_calls"] = message.tool_calls
                messages.append(assistant_msg)
                
                if message.content:
                    console.print(Markdown(message.content))
        
        except KeyboardInterrupt:
            raise
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            continue


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
