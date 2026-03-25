# Book Data Maker - Copilot Instructions

## Project Overview
**Version**: 4.1.2 (`pyproject.toml`) · **Python**: 3.10–3.13 · **License**: MIT

A Python CLI tool for extracting text from documents using DeepSeek OCR and generating multi-turn conversation datasets via parallel LLM threads. The system uses a two-stage pipeline: (1) OCR extraction creates page-based text files, (2) Multiple LLM threads navigate the extracted content using tools to generate multi-turn conversations stored in SQLite, then exported to Parquet/JSONL/CSV.

## Architecture

### Two-Stage Pipeline
1. **Stage 1 - Extract**: `bookdatamaker extract` → OCR text extraction → `page_XXX/result.mmd` files + `combined.txt`
2. **Stage 2 - Generate**: `bookdatamaker generate` → Parallel LLM threads with tool calling → SQLite → `bookdatamaker export-dataset` → Parquet/JSONL/CSV

### Source Layout
```
src/bookdatamaker/
├── cli.py                          # Click CLI (extract, generate, export-dataset, mcp-server)
├── __init__.py                     # Package version (__version__)
├── dataset/
│   ├── dataset_manager.py          # SQLite storage, duplicate detection, export, thread state
│   └── builder.py                  # Dataset builder utilities
├── llm/
│   ├── parallel_generator.py       # Multi-threaded generation engine (API + vLLM modes)
│   ├── prompt_builder.py           # Template-based system prompt construction
│   ├── message_utils.py            # Serialization, pruning, sanitization, <think> extraction
│   ├── minimax_mcp.py              # MiniMax MCP proxy (web_search, understand_image)
│   └── search1api_mcp.py           # Search1API MCP proxy (search, news, crawl)
├── mcp/
│   └── server.py                   # Standalone MCP server (stdio transport)
├── ocr/
│   ├── extractor.py                # Dual-mode OCR (API/local, OCR-1/OCR-2)
│   └── document_parser.py          # PDF/EPUB/PPTX → images
├── prompts/
│   ├── base_system_prompt.md       # Core template with {{placeholders}}
│   ├── minimax_guidance.md         # MiniMax web search guidance
│   ├── search1api_guidance.md      # Search1API guidance
│   ├── combined_search_guidance.md # Dual-provider guidance
│   └── image_workflow.md           # Image analysis workflow
├── tools/
│   ├── registry.py                 # Centralized tool schemas (OpenAI + MCP formats)
│   ├── page_tools.py               # Page navigation tool dispatcher
│   ├── image_tools.py              # Image listing and base64 encoding
│   └── submission_tools.py         # Validation, analytics, page access summary
└── utils/
    ├── page_manager.py             # In-memory page storage with indexing
    └── status.py                   # StatusIndicator context manager
```

### Core Components
- **OCRExtractor** (`ocr/extractor.py`): Dual-mode (API/local) text extraction with DeepSeek-OCR. Supports OCR-1 and OCR-2 via `ocr_version` parameter (default: "2"). Version-specific config in `_VERSION_CONFIG` class dict. Requires `transformers==4.46.3` for local mode.
- **DocumentParser** (`ocr/document_parser.py`): Converts PDF (PyMuPDF), EPUB (ebooklib), and PPTX (python-pptx) to images for OCR. Supports streaming (`iter_pdf_images`) for memory efficiency.
- **PageManager** (`utils/page_manager.py`): In-memory document storage with line/column/paragraph indexing. Loads from `page_XXX/` directories or `combined.txt` with `[PAGE_XXX]` markers.
- **ToolRegistry** (`tools/registry.py`): Single source of truth for all tool schemas. Generates both OpenAI function-calling format (`build_openai_tool_defs`) and MCP format (`build_mcp_tool_specs`). Shared `TOOL_DESCRIPTIONS` dict prevents description drift.
- **PageTools** (`tools/page_tools.py`): Unified `execute_page_tool()` dispatcher for navigation tools. Returns structured `{"ok": bool, "pages_touched": [...]}` results.
- **ImageTools** (`tools/image_tools.py`): `list_page_images()` enumerates full-page and cropped images; `get_image_data_url()` returns base64 data URLs with path traversal protection.
- **SubmissionTools** (`tools/submission_tools.py`): `validate_dataset_messages()` enforces format rules; `build_page_access_summary()` computes per-page submission analytics.
- **PromptBuilder** (`llm/prompt_builder.py`): Template-based system prompt construction. Loads `.md` templates from `prompts/` with LRU caching. Replaces `{{placeholders}}` with thread metadata and conditional tool sections.
- **MessageUtils** (`llm/message_utils.py`): `serialize_messages()` for checkpoint persistence, `extract_think()` for `<think>` tag extraction, `sanitize_tool_pairs()` for tool-call integrity, `safe_prune_messages()` for history management.
- **MCPServer** (`mcp/server.py`): Standalone MCP server with stdio transport. Integrates PageManager, DatasetManager, and image tools. Supports both page-based and legacy paragraph-based navigation.
- **ParallelDatasetGenerator** (`llm/parallel_generator.py`): Spawns N threads at distributed positions. Each thread uses OpenAI client + tool calling. Supports resume from checkpoint. Integrates MiniMax and Search1API MCP proxies.
- **MiniMaxMCP** (`llm/minimax_mcp.py`): MCP proxy for MiniMax API (web_search, understand_image). Auto-enabled when model name contains "minimax".
- **Search1APIMCP** (`llm/search1api_mcp.py`): Remote MCP proxy via HTTP Streamable transport to `https://mcp.search1api.com/mcp`. Thread-safe with per-thread sessions and lazy initialization. Tools prefixed with `search1api_`.
- **DatasetManager** (`dataset/dataset_manager.py`): SQLite storage with WAL mode, duplicate detection (rapidfuzz, threshold 85%), thread state tracking, and multi-format export via HuggingFace datasets. Retry logic with exponential backoff for database locks.

### Key Data Flows
1. **Extraction Flow**: Document → DocumentParser (PDF/EPUB/PPTX) → images → OCRExtractor (API/local) → `page_XXX/result.mmd` + `combined.txt`
2. **Generation Flow**: `page_XXX/` dirs → PageManager → Tool Registry → LLM threads → submit_dataset calls → DatasetManager → SQLite
3. **Export Flow**: SQLite → DatasetManager.export_X() → Parquet/JSONL/CSV/JSON

### Tool Architecture
Tools are defined once in `tools/registry.py` and consumed by two surfaces:

| Surface | Builder Function | Format |
|---------|-----------------|--------|
| Generator (OpenAI API) | `build_openai_tool_defs()` | OpenAI function calling |
| MCP Server | `build_mcp_tool_specs()` | MCP Tool specs |

**Core tools** (always available):
- `submit_dataset` — Submit multi-turn conversation
- `exit` — End session after target reached
- `get_current_page`, `next_page`, `previous_page`, `jump_to_page` — Navigation
- `get_page_context`, `get_page_range` — Multi-page retrieval (max 5 pages)
- `search_text` — Full-document search with `max_results` (OpenAI default: 20, MCP default: 100)
- `get_page_access_summary` — Per-page submission analytics

**Conditional tools** (enabled by flags/keys):
- `list_page_images`, `get_image` — When `extracted_dir` has image assets
- `minimax_web_search`, `minimax_understand_image` — When `--minimax-mcp-key` provided
- `search1api_search`, `search1api_news`, `search1api_crawl` — When `--search1api-key` provided

### Prompt System
System prompts are built from Markdown templates in `src/bookdatamaker/prompts/`:

| Template | Purpose | Loaded When |
|----------|---------|-------------|
| `base_system_prompt.md` | Core instructions with `{{placeholders}}` | Always |
| `minimax_guidance.md` | MiniMax web search usage guidance | MiniMax MCP enabled |
| `search1api_guidance.md` | Search1API usage guidance | Search1API enabled |
| `combined_search_guidance.md` | Dual-provider guidance | Both enabled |
| `image_workflow.md` | Image analysis workflow | Images + MiniMax |

**Placeholders**: `{{START_PAGE}}`, `{{THREAD_ID}}`, `{{TOTAL_PAGES}}`, `{{TARGET_COUNT}}`, `{{IMAGE_TOOLS_LINE}}`, `{{MINIMAX_TOOL_LINES}}`, `{{SEARCH1API_TOOL_LINES}}`, `{{SEARCH_GUIDANCE_SECTION}}`, `{{IMAGE_WORKFLOW_SECTION}}`, `{{CUSTOM_PROMPT_SECTION}}`

## Development Conventions

### Code Style
- **Type hints**: Required for all functions (enforced by mypy). Use `Optional[]`, `list[]`, `dict[]` from `typing` or builtins (Python 3.10+).
- **Async/await**: Required for I/O operations (OCR API calls, file operations in extractors). Use `asyncio.run()` in CLI commands.
- **Error handling**: Wrap external service calls (API, vLLM) in try-except. Use custom exceptions like `DuplicateEntryError`.
- **Formatting**: Black (line-length 100), Ruff for linting. Config in `pyproject.toml`.

### Module Organization
- Each module is self-contained with clear responsibilities
- CLI commands in `cli.py` delegate to async helpers (`_extract_async`, `_generate_async`)
- Tool schemas centralized in `tools/registry.py` — never define tool schemas inline
- Tool execution logic in `tools/page_tools.py`, `tools/image_tools.py`, `tools/submission_tools.py`
- Prompt templates in `prompts/` directory — never hardcode prompt text in Python
- Message utilities in `llm/message_utils.py` — serialization, pruning, sanitization
- Util classes (`PageManager`, `StatusIndicator`) are reusable across commands
- MCP server is standalone but integrates with PageManager and DatasetManager

### Critical Dependencies
- `transformers==4.46.3`: Exact version required for DeepSeek-OCR compatibility (warning shown if mismatch)
- `mcp`: Model Context Protocol for LLM tool integration
- `openai`: Used for both OpenAI API and vLLM endpoints (OpenAI-compatible)
- `rapidfuzz`: Fuzzy matching for duplicate detection (default threshold: 85%)
- `pyarrow` + `datasets`: Parquet export with zstd compression via HuggingFace datasets
- `click`: CLI framework
- `rich`: Console output formatting
- `httpx`: HTTP client for API calls
- `Pillow`: Image processing for OCR and image tools
- `PyMuPDF`, `ebooklib`, `python-pptx`: Document format parsing

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

# EPUB/PPTX (always plain text, no OCR needed)
bookdatamaker extract book.epub -o ./extracted
bookdatamaker extract slides.pptx -o ./extracted
```
Output: `./extracted/page_001/result.mmd`, ..., `./extracted/combined.txt`

### Generate Dataset (Stage 2)
```bash
# API mode (OpenAI/DeepSeek compatible)
bookdatamaker generate ./extracted -d dataset.db \
  --distribution "10,10,20,30,20,10" --datasets-per-thread 20 \
  --openai-api-url https://api.openai.com/v1 --model gpt-4

# With web search (MiniMax)
bookdatamaker generate ./extracted -d dataset.db \
  --model MiniMax-Text-01 --minimax-mcp-key YOUR_KEY

# With web search (Search1API, works with any model)
bookdatamaker generate ./extracted -d dataset.db \
  --model gpt-4 --search1api-key YOUR_KEY

# With message history limit (prevents token overflow)
bookdatamaker generate ./extracted -d dataset.db --max-messages 50

# With API rate limiting
bookdatamaker generate ./extracted -d dataset.db --api-delay 0.5

# vLLM mode (local model)
bookdatamaker generate ./extracted -d dataset.db --mode vllm \
  --vllm-model-path meta-llama/Llama-3-8B-Instruct \
  --tool-call-parser hermes --tensor-parallel-size 2

# Resume from checkpoint (automatic prompt if incomplete threads detected)
bookdatamaker generate ./extracted -d dataset.db  # Auto-detects incomplete threads
```

### Export Dataset
```bash
bookdatamaker export-dataset dataset.db -o output.parquet              # Parquet (default, zstd)
bookdatamaker export-dataset dataset.db -o output.parquet -c snappy    # Parquet with snappy
bookdatamaker export-dataset dataset.db -o output.jsonl -f jsonl
bookdatamaker export-dataset dataset.db -o output.csv -f csv
bookdatamaker export-dataset dataset.db -o output.json -f json
```

### MCP Server (Manual Testing)
```bash
bookdatamaker mcp-server ./extracted -d dataset.db  # Starts stdio server for LLM integration
```

## Important Patterns

### Multi-Turn Conversation Format
Dataset stores multi-turn conversations (not single Q&A pairs).
- `submit_dataset` accepts `messages: list[str]` (string array)
- Must alternate between user and assistant messages
- Must start with user message, end with assistant message
- Minimum 2 messages (1 turn), can be 4, 6, 8+ for multi-turn
- Example: `["What is X?", "X is...", "Can you explain more?", "Sure, X also..."]`
- Validation: `submission_tools.validate_dataset_messages()` checks format
- Storage: JSON array in SQLite `messages` column with `{"role": "user/assistant", "content": "..."}`

### Position Distribution
`--distribution "10,10,20,30,20,10"` defines:
- **Number of threads**: 6 threads (comma-separated count)
- **Starting positions**: Cumulative percentages → Thread 0 at 10%, Thread 1 at 20%, Thread 2 at 40%, etc.
- Implementation: `ParallelDatasetGenerator.calculate_positions()` converts percentages to page numbers

### Checkpoint Resume
- Thread state stored in `thread_state` table (position, messages, submitted_count, target_count, status)
- On `generate` command, checks for incomplete threads (`status != 'completed'`)
- Prompts user to resume or clear checkpoint
- Serializes OpenAI tool_calls to JSON for storage via `message_utils.serialize_messages()`

### Message History Management
- `--max-messages` CLI option limits conversation history length
- When exceeded, `message_utils.safe_prune_messages(messages, keep_last=10)` is called
- Preserves system message + last N messages
- `sanitize_tool_pairs()` ensures no orphaned tool_call/tool_result pairs
- Prevents token overflow in long generation sessions

### Tool Execution Flow (Generator)
1. `registry.build_openai_tool_defs()` builds tool list (with optional MiniMax/Search1API extras)
2. LLM calls OpenAI API with `tools` parameter and `tool_choice="auto"`
3. Response tool_calls are dispatched:
   - `submit_dataset` → `submission_tools.validate_dataset_messages()` → `DatasetManager.add_entry()`
   - `exit` → Check target reached, save final state
   - Page tools → `page_tools.execute_page_tool()` → PageManager methods
   - Image tools → `image_tools.list_page_images()` or `get_image_data_url()`
   - MiniMax tools → `MiniMaxMCP.call_tool()`
   - Search1API tools → `Search1APIMCP.call_tool()` (identified by `search1api_` prefix)
4. Tool results formatted as `{"tool_call_id": ..., "role": "tool", "content": json.dumps(result)}`

### Web Search Integration
Two optional web search providers, can be used independently or together:

| Provider | Flag | Auto-detect | Tools |
|----------|------|-------------|-------|
| MiniMax | `--minimax-mcp-key` | Yes, if model name contains "minimax" | `minimax_web_search`, `minimax_understand_image` |
| Search1API | `--search1api-key` | No | `search1api_search`, `search1api_news`, `search1api_crawl` |

- MiniMax: Uses MCP protocol, provides both web search and image understanding
- Search1API: Remote MCP via HTTP Streamable transport, thread-safe with per-thread sessions
- When both enabled, `combined_search_guidance.md` template is used

### Duplicate Detection
- `DatasetManager.find_similar_entry()` uses rapidfuzz ratio matching
- Checks entire messages array (concatenated role:content strings)
- Raises `DuplicateEntryError` if similarity > threshold (default 85%)
- Used in `add_entry()` to prevent duplicate conversations
- Generator tracks and logs duplicate warnings per thread

### Page Manager Loading
Two modes:
1. **Directory mode**: `PageManager.from_directory(extracted_dir)` loads from `page_XXX/result.mmd` files
2. **Combined file mode**: `PageManager.from_combined_file(combined.txt)` parses `[PAGE_XXX]` markers

Builds indexes for:
- `line_to_page`: Global line → (page_num, local_line)
- `page_line_ranges`: Page → (start_line, end_line)
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

## Testing

### Test Files
| Test File | Covers |
|-----------|--------|
| `test_ocr.py` | OCR API and local modes |
| `test_dataset.py` | SQLite storage, duplicate detection, export |
| `test_mcp.py` | MCP server tool execution |
| `test_paragraph_indexing.py` | Line/paragraph index building |
| `test_message_utils.py` | Message serialization, pruning, sanitization |
| `test_prompt_builder.py` | Prompt generation with various tool configurations |
| `test_tools_registry.py` | OpenAI and MCP tool definition generation |
| `test_page_tools.py` | Page navigation tool execution |
| `test_image_tools.py` | Image listing and base64 conversion |
| `test_submission_tools.py` | Validation and page analytics |
| `test_resume_reliability.py` | Checkpoint restore with edge cases |

### Testing Conventions
- Tests use temporary directories for file operations
- Mock OCR API calls to avoid external dependencies
- Test both API and local modes in `test_ocr.py`
- Use `pytest-asyncio` for async test functions
- Verbose output with short tracebacks (configured in `pytest.ini`)

## Version Constraints
- **Python**: 3.10–3.13 (`requires-python = ">=3.10,<=3.13"`)
- **transformers**: Must be 4.46.3 (checked in `OCRExtractor._init_local_model()`)
- **CUDA**: Required for GPU acceleration in local mode

## Common Pitfalls
- **Forgetting `transformers==4.46.3`**: DeepSeek-OCR (both v1 and v2) will fail with other versions
- **Missing `--tool-call-parser`**: Required for vLLM mode, causes cryptic errors
- **Distribution mismatch**: Number of distribution values determines thread count, not a `--threads` flag
- **OOM in vLLM**: Reduce `--max-model-len` if GPU memory errors occur
- **Checkpoint conflicts**: Clear with `DatasetManager.clear_thread_states()` if resuming fails
- **Invalid messages array**: Must be even length, start with user, end with assistant (validated in `submission_tools.validate_dataset_messages()`)
- **OCR version mismatch**: Ensure vLLM server model matches `--ocr-version` (OCR-2 model with `--ocr-version 2`)
- **Token overflow**: Use `--max-messages` to limit conversation history for long sessions
- **search_text max_results inconsistency**: OpenAI surface defaults to 20, MCP surface defaults to 100
- **Defining tools inline**: Always use `tools/registry.py` — never define tool schemas in generator or MCP server directly
- **Hardcoding prompts**: Always use templates in `prompts/` directory — never embed prompt text in Python code
