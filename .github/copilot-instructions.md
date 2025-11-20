# Book Data Maker - Copilot Instructions

## Project Overview
A Python CLI tool for extracting text from documents using DeepSeek OCR and generating multi-turn conversation datasets via parallel LLM threads. The system uses a two-stage pipeline: (1) OCR extraction creates page-based text files, (2) Multiple LLM threads navigate the extracted content using MCP tools to generate multi-turn conversations stored in SQLite, then exported to Parquet/JSONL/CSV.

## Architecture

### Two-Stage Pipeline
1. **Stage 1 - Extract**: `bookdatamaker extract` → OCR text extraction → `page_XXX/result.mmd` files + `combined.txt`
2. **Stage 2 - Generate**: `bookdatamaker generate` → Parallel LLM threads with MCP navigation → SQLite → `bookdatamaker export-dataset` → Parquet/JSONL/CSV

### Core Components
- **OCRExtractor** (`ocr/extractor.py`): Dual-mode (API/local) text extraction with DeepSeek-OCR. Requires `transformers==4.46.3` for local mode.
- **PageManager** (`utils/page_manager.py`): In-memory document storage with line/column/paragraph indexing. Loads from `page_XXX/` directories or `combined.txt` with `[PAGE_XXX]` markers.
- **MCPServer** (`mcp/server.py`): Provides 15+ navigation tools (pages, lines, paragraphs, search, submit_dataset) for LLMs. Uses stdio transport.
- **ParallelDatasetGenerator** (`llm/parallel_generator.py`): Spawns N threads at distributed positions. Each thread uses OpenAI client + MCP tools. Supports resume from checkpoint.
- **DatasetManager** (`dataset/dataset_manager.py`): SQLite storage with duplicate detection (rapidfuzz), thread state tracking, and multi-format export. Stores conversations as JSON arrays of user/assistant messages.

### Key Data Flows
1. **Extraction Flow**: Document → DocumentParser (PDF/EPUB) → images → OCRExtractor (API/local) → `page_XXX/result.mmd` + `combined.txt`
2. **Generation Flow**: `page_XXX/` dirs → PageManager → MCPServer tools → LLM threads → submit_dataset calls → DatasetManager → SQLite
3. **Export Flow**: SQLite → DatasetManager.export_X() → Parquet/JSONL/CSV/JSON

## Development Conventions

### Code Style
- **Type hints**: Required for all functions (enforced by mypy). Use `Optional[]`, `List[]`, `Dict[]` from `typing`.
- **Async/await**: Required for I/O operations (OCR API calls, file operations in extractors). Use `asyncio.run()` in CLI commands.
- **Error handling**: Wrap external service calls (API, vLLM) in try-except. Use custom exceptions like `DuplicateEntryError`.
- **Formatting**: Black (line-length 100), Ruff for linting. Config in `pyproject.toml`.

### Module Organization
- Each module is self-contained with clear responsibilities
- CLI commands in `cli.py` delegate to async helpers (`_extract_async`, `_generate_async`)
- Util classes (`PageManager`, `StatusIndicator`) are reusable across commands
- MCP server is standalone but integrates with PageManager and DatasetManager

### Critical Dependencies
- `transformers==4.46.3`: Exact version required for DeepSeek-OCR compatibility (warning shown if mismatch)
- `mcp`: Model Context Protocol for LLM tool integration
- `openai`: Used for both OpenAI API and vLLM endpoints (OpenAI-compatible)
- `rapidfuzz`: Fuzzy matching for duplicate detection (default threshold: 85%)
- `pyarrow`: Parquet export with zstd compression

## Common Workflows

### Running Tests
```bash
pytest tests/              # All tests
pytest tests/test_ocr.py   # Specific test
pytest -v --tb=short       # Verbose with short tracebacks (default in pytest.ini)
```

### Extract Text (Stage 1)
```bash
# API mode (vLLM server required)
bookdatamaker extract book.pdf --deepseek-api-url http://localhost:8000/v1 -o ./extracted

# Local mode (transformers)
bookdatamaker extract book.pdf --mode local --batch-size 8 --device cuda -o ./extracted

# Plain text (no OCR)
bookdatamaker extract book.pdf --plain-text -o ./extracted
```
Output: `./extracted/page_001/result.mmd`, ..., `./extracted/combined.txt`

### Generate Dataset (Stage 2)
```bash
# API mode (OpenAI/DeepSeek compatible)
bookdatamaker generate ./extracted -d dataset.db \
  --distribution "10,10,20,30,20,10" --datasets-per-thread 20 \
  --openai-api-url https://api.openai.com/v1 --model gpt-4

# vLLM mode (local model)
bookdatamaker generate ./extracted -d dataset.db --mode vllm \
  --vllm-model-path meta-llama/Llama-3-8B-Instruct \
  --tool-call-parser hermes --tensor-parallel-size 2

# Resume from checkpoint (automatic prompt if incomplete threads detected)
bookdatamaker generate ./extracted -d dataset.db  # Auto-detects incomplete threads
```

### Export Dataset
```bash
bookdatamaker export-dataset dataset.db -o output.parquet       # Parquet (default)
bookdatamaker export-dataset dataset.db -o output.jsonl -f jsonl
bookdatamaker export-dataset dataset.db -o output.csv -f csv
```

### MCP Server (Manual Testing)
```bash
bookdatamaker mcp-server ./extracted -d dataset.db  # Starts stdio server for LLM integration
```

## Important Patterns

### Multi-Turn Conversation Format
**NEW**: Dataset now stores conversations instead of single Q&A pairs.
- `submit_dataset` accepts `messages: List[str]` (string array)
- Must alternate between user and assistant messages
- Must start with user message, end with assistant message
- Minimum 2 messages (1 turn), can be 4, 6, 8+ for multi-turn
- Example: `["What is X?", "X is...", "Can you explain more?", "Sure, X also..."]`
- Storage: JSON array in SQLite `messages` column with `{"role": "user/assistant", "content": "..."}`

### Position Distribution
`--distribution "10,10,20,30,20,10"` defines:
- **Number of threads**: 6 threads (comma-separated count)
- **Starting positions**: Thread 0 starts at 10% of document, Thread 1 at 20%, Thread 2 at 40%, etc.
- Implementation: `ParallelDatasetGenerator.calculate_positions()` converts percentages to page numbers

### Checkpoint Resume
- Thread state stored in `thread_state` table (position, messages, submitted_count)
- On `generate` command, checks for incomplete threads (`status != 'completed'`)
- Prompts user to resume or clear checkpoint
- Serializes OpenAI tool_calls to JSON for storage (`_serialize_messages()`)

### MCP Tool Integration
- `ParallelDatasetGenerator._get_mcp_tools()` defines 15+ tools (get_page, search_text, submit_dataset, etc.)
- LLM calls OpenAI API with `tools` parameter
- Tool results formatted as `{"tool_call_id": ..., "role": "tool", "content": json.dumps(result)}`
- `submit_dataset` is terminal action that increments thread's submitted_count

### Duplicate Detection
- `DatasetManager.find_similar_entry()` uses rapidfuzz ratio matching
- Checks entire messages array (concatenated role:content strings)
- Raises `DuplicateEntryError` if similarity > threshold (default 85%)
- Used in `add_entry()` to prevent duplicate conversations

### Page Manager Loading
Two modes:
1. **Directory mode**: `PageManager.from_directory(extracted_dir)` loads from `page_XXX/result.mmd` files
2. **Combined file mode**: `PageManager.from_combined_file(combined.txt)` parses `[PAGE_XXX]` markers

Builds indexes for:
- `line_to_page`: Global line → (page_num, local_line)
- `line_to_paragraph`: Line → paragraph number
- `paragraph_line_ranges`: Paragraph → (start_line, end_line)

### Status Indicators
Use `StatusIndicator` context manager for progress tracking:
```python
from bookdatamaker.utils import StatusIndicator

with StatusIndicator() as status:
    status.print_info("Starting extraction...")
    # ... work ...
    status.print_success("Extraction complete")
```

## Testing Considerations
- Tests use temporary directories for file operations
- Mock OCR API calls to avoid external dependencies
- Test both API and local modes in `test_ocr.py`
- Paragraph indexing tests in `test_paragraph_indexing.py` validate PageManager logic
- Use `pytest-asyncio` for async test functions

## Version Constraints
- **Python**: 3.10-3.12 (3.13 not supported due to vLLM incompatibility)
- **transformers**: Must be 4.46.3 (checked in `OCRExtractor._init_local_model()`)
- **CUDA**: Required for GPU acceleration in local mode

## Common Pitfalls
- **Forgetting `transformers==4.46.3`**: DeepSeek-OCR will fail with other versions
- **Missing `--tool-call-parser`**: Required for vLLM mode, causes cryptic errors
- **Distribution mismatch**: Number of distribution values determines thread count, not `--threads`
- **OOM in vLLM**: Reduce `--max-model-len` if GPU memory errors occur
- **Checkpoint conflicts**: Clear with `DatasetManager.clear_thread_states()` if resuming fails
- **Invalid messages array**: Must be even length, start with user, end with assistant (validated in `add_entry()`)
