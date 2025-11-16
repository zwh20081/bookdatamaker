# Book Data Maker

A powerful CLI tool for extracting text from documents using DeepSeek OCR and generating high-quality datasets with LLM assistance.

## Table of Contents

### 🚀 Getting Started
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)

### 📖 User Guide
- [Extract Text (Stage 1)](#extract-text-stage-1)
- [Generate Dataset (Stage 2)](#generate-dataset-stage-2)
- [Export Dataset](#export-dataset)

### 🔧 Advanced
- [Position Distribution](#position-distribution)
- [Performance Tuning](#performance-tuning)
- [Interactive Chat](#interactive-chat)
- [MCP Server](#mcp-server)

### 📚 Reference
- [Command Reference](#command-reference)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

---

## Features

- 📄 **Multi-Format Support**: PDF, EPUB, and images
- 🏠 **Self-Hosted OCR**: Local transformers for DeepSeek-OCR (no API costs)
- 🤖 **Parallel Generation**: Multiple LLM threads explore documents simultaneously
- 🎯 **Smart Distribution**: Control thread starting positions
- 💾 **SQLite Storage**: Real-time dataset storage with flexible export
- 📊 **Multiple Formats**: JSONL, Parquet, CSV, JSON
- 🌐 **Flexible Modes**: API or self-hosted for both stages
- 📈 **Progress Tracking**: Real-time progress bars

## Installation

### From PyPI (Recommended)

```bash
pip install bookdatamaker
```

### From Source

```bash
git clone https://github.com/yourusername/bookdatamaker.git
cd bookdatamaker
pip install -r requirements.txt
pip install -e .
```

### Optional: Local Inference Support

```bash
# For self-hosted OCR and LLM generation
pip install bookdatamaker[local]  # From PyPI
# OR
pip install -e ".[local]"  # From source - installs transformers==4.46.3, torch, flash-attn, etc.
```

**Note**: The project requires `transformers==4.46.3` for optimal compatibility with DeepSeek-OCR. A warning will be displayed if a different version is detected.

### System Requirements

**For API Mode:**
- Python 3.10+
- API keys (OpenAI, DeepSeek, etc.)

**For Local Mode:**
- Python 3.10-3.12 (3.13 not supported due to vLLM compatibility)
- NVIDIA GPU with CUDA support (or CPU, though slower)
- 16GB+ VRAM recommended for GPU
- transformers==4.46.3
- Linux or WSL2 (recommended)

---

## Quick Start

### Prerequisites

```bash
# Set API keys (choose one based on your mode)
export OPENAI_API_KEY=your_openai_key        # For API mode
export DEEPSEEK_API_KEY=your_deepseek_key    # For API OCR mode
```

### Option 1: API Mode (Fastest Setup)

```bash
# 1. Install
pip install bookdatamaker

# 2. Extract → Generate → Export
bookdatamaker extract book.pdf -o ./extracted
bookdatamaker generate ./extracted -d dataset.db --distribution "10,10,20,30,20,10"
bookdatamaker export-dataset dataset.db -o output.parquet
```

### Option 2: Self-Hosted Mode (Free, Private)

```bash
# 1. Install with local dependencies
pip install bookdatamaker[local]

# 2. Extract with local OCR
bookdatamaker extract book.pdf --mode local --batch-size 8 -o ./extracted

# 3. Generate with vLLM
bookdatamaker generate ./extracted \
  --mode vllm \
  --vllm-model-path meta-llama/Llama-3-8B-Instruct \
  --distribution "25,25,25,25" \
  -d dataset.db

# 4. Export
bookdatamaker export-dataset dataset.db -o output.parquet
```

---

## System Requirements

**For API Mode:**
- Python 3.10+
- API keys (OpenAI, DeepSeek, etc.)

**For Local Mode:**
- Python 3.10-3.12 (3.13 not supported due to vLLM compatibility)
- NVIDIA GPU with CUDA support (or CPU, though slower)
- 16GB+ VRAM recommended for GPU
- transformers==4.46.3
- Linux or WSL2 (recommended)

---

## Extract Text (Stage 1)

Extract text from documents using DeepSeek OCR.

### Supported Formats

- **PDF**: Text extraction or OCR from rendered pages
- **EPUB**: E-book text extraction
- **Images**: JPG, PNG, BMP, TIFF, WebP

### API Mode

**Note**: DeepSeek does not provide an official OCR API. You need to self-host DeepSeek-OCR using vLLM.

#### Setup vLLM OCR Server

Follow the [vLLM DeepSeek-OCR recipe](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html) to set up your server


#### Use the API

Once your vLLM server is running:

```bash
# Basic usage (default: http://localhost:8000/v1)
bookdatamaker extract book.pdf -o ./extracted

# Custom vLLM endpoint
bookdatamaker extract book.pdf \
  --deepseek-api-url http://your-server:8000/v1 \
  -o ./extracted

# Adjust concurrency for faster processing
bookdatamaker extract book.pdf \
  --api-concurrency 8 \
  -o ./extracted
```

**Performance Options:**
- `--api-concurrency N`: Number of concurrent API requests (default: 4)
  - Higher values = faster processing (if your server can handle it)
  - Adjust based on your vLLM server capacity and network bandwidth
  - Example: 8-16 for powerful servers, 2-4 for smaller setups

### Local Mode (Transformers)

Use local transformers model for OCR (DeepSeek-OCR, no API calls):

```bash
# Basic usage - uses transformers AutoModel with flash_attention_2
bookdatamaker extract book.pdf --mode local -o ./extracted

# With custom batch size (adjust based on GPU memory)
bookdatamaker extract book.pdf --mode local --batch-size 12 -o ./extracted

# Use CPU instead of GPU
bookdatamaker extract book.pdf --mode local --device cpu -o ./extracted

# Use specific GPU
bookdatamaker extract book.pdf --mode local --device cuda:1 -o ./extracted

# Process directory of images
bookdatamaker extract ./images/ --mode local -o ./extracted
```

**Performance Options:**
- `--batch-size N`: Number of images to process in parallel (default: 8)
  - Higher values = faster processing but more GPU memory
  - Adjust based on available VRAM
  - Example: 4 for 8GB VRAM, 8-16 for 24GB+ VRAM

**Device Options:**
- `cuda` (default): Use default CUDA GPU
- `cuda:0`, `cuda:1`, etc.: Use specific GPU
- `cpu`: Use CPU (slower, no GPU required)
- `xpu`: Use Intel XPU

### Plain Text Mode (No OCR)

For PDF with embedded text, skip OCR and extract text directly (much faster):

```bash
# Extract plain text from PDF without OCR
bookdatamaker extract book.pdf --plain-text -o ./extracted
```

**Note**: EPUB files are **automatically extracted as plain text** (no OCR needed, no `--plain-text` flag required):

```bash
# EPUB always uses plain text extraction
bookdatamaker extract book.epub -o ./extracted
```

**When to use `--plain-text` (for PDF):**
- ✅ PDF with embedded text (e.g., born-digital documents)
- ✅ Fast extraction without GPU/API requirements
- ✅ Text-only documents

**When NOT to use `--plain-text`:**
- ❌ Scanned PDFs (images of text)
- ❌ PDFs with complex layouts requiring OCR
- ❌ Documents where text extraction quality is poor

### Output Structure

```
./extracted/
├── page_001/
│   ├── page_001.png      # Page image
│   └── result.mmd        # Extracted text in markdown
├── page_002/
│   ├── page_002.png
│   └── result.mmd
└── ...
```

**Note**: Each page is stored in its own subdirectory with the extracted text in `result.mmd` format.

---

## Generate Dataset (Stage 2)

Generate Q&A datasets using parallel LLM threads with **page-based navigation**.

### Navigation Model

The system uses **page navigation**:
- LLM threads navigate through document pages
- Tools available: `get_current_page`, `next_page`, `previous_page`, `jump_to_page`, `get_page_context`
- Each thread starts at a specific page based on distribution
- Threads can move forward/backward through pages to explore content

### Checkpoint & Resume

The generation process **automatically saves checkpoints** to the database:
- Thread state is saved after each successful Q&A submission
- If interrupted (Ctrl+C, crash, etc.), simply rerun the same command
- You'll be prompted to resume from checkpoint or start fresh

```bash
# First run (interrupted at 50%)
bookdatamaker generate ./extracted -d dataset.db --distribution "25,25,25,25"
# ^C (interrupted)

# Resume from checkpoint
bookdatamaker generate ./extracted -d dataset.db --distribution "25,25,25,25"
# ⚠️  Found 4 incomplete thread(s) in database:
#   Thread 0: 8/20 pairs, last updated 2024-01-15 10:30:45
#   Thread 1: 10/20 pairs, last updated 2024-01-15 10:30:48
#   Thread 2: 12/20 pairs, last updated 2024-01-15 10:30:50
#   Thread 3: 7/20 pairs, last updated 2024-01-15 10:30:43
# 
# Do you want to resume from checkpoint? [Y/n]: y
# ✓ Resuming from checkpoint...
```

**Features:**
- 💾 Automatic checkpoint after each Q&A pair submission
- 🔄 Resume from last position in document
- 💬 Preserves conversation history
- 🎯 Tracks progress per thread

### Basic Usage

```bash
# 6 threads (from distribution), 20 Q&A pairs per thread
bookdatamaker generate ./extracted \
  -d dataset.db \
  --distribution "10,10,20,30,20,10" \
  --datasets-per-thread 20
```
**Key Concept**: Thread count is determined by the number of comma-separated values in `--distribution`.

### API Mode Examples

```bash
# OpenAI/Azure
bookdatamaker generate ./extracted \
  -d dataset.db \
  --openai-api-url https://api.openai.com/v1 \
  --model gpt-4 \
  --distribution "10,10,20,30,20,10"

# Custom API endpoint
bookdatamaker generate ./extracted \
  --openai-api-url http://localhost:8000/v1 \
  --model your-model-name \
  --distribution "25,25,25,25"
```

### vLLM Direct Mode (Self-Hosted)

Use vLLM directly without API server:

```bash
# Single GPU
bookdatamaker generate ./extracted \
  --mode vllm \
  --vllm-model-path meta-llama/Llama-3-8B-Instruct \
  --distribution "25,25,25,25" \
  -d dataset.db

# Multi-GPU (4 GPUs, 6 threads)
bookdatamaker generate ./extracted \
  --mode vllm \
  --vllm-model-path meta-llama/Llama-3-70B-Instruct \
  --tensor-parallel-size 4 \
  --distribution "10,10,20,30,20,10" \
  -d dataset.db
```

### Custom Prompts

Add specific instructions to guide LLM behavior:

```bash
# Language specification
bookdatamaker generate ./extracted \
  --custom-prompt "Generate all Q&A in Chinese with simplified characters"

# Format specification
bookdatamaker generate ./extracted \
  --custom-prompt "Questions should be multiple-choice with 4 options"

# Multiple requirements
bookdatamaker generate ./extracted \
  --custom-prompt "Requirements:
1. Generate questions in English
2. Focus on practical applications
3. Include code examples
4. Answer length: 50-150 words
5. Difficulty: intermediate"
```

### Message History Management

Control conversation history to prevent token overflow:

```bash
# Limit conversation to 50 messages (keeps system prompt + last 10 when exceeded)
bookdatamaker generate ./extracted \
  --max-messages 50 \
  -d dataset.db

# For models with limited context windows
bookdatamaker generate ./extracted \
  --max-messages 30 \
  --model gpt-3.5-turbo
```

**How it works:**
- When message count exceeds `--max-messages`, history is pruned automatically
- System prompt is always preserved
- Last 10 messages are kept for continuity
- Prevents token overflow errors during long generation sessions
- Useful for models with limited context windows (e.g., 4K, 8K tokens)

---

## Export Dataset

Export from SQLite database to your preferred format:

```bash
# Parquet (recommended for data analysis, default: zstd compression)
bookdatamaker export-dataset dataset.db -o output.parquet

# Parquet with different compression methods
bookdatamaker export-dataset dataset.db -o output.parquet -c snappy  # Faster, larger files
bookdatamaker export-dataset dataset.db -o output.parquet -c gzip    # Smaller, slower
bookdatamaker export-dataset dataset.db -o output.parquet -c brotli  # Best compression
bookdatamaker export-dataset dataset.db -o output.parquet -c none    # No compression

# JSON Lines (easy to stream)
bookdatamaker export-dataset dataset.db -o output.jsonl -f jsonl

# CSV (Excel-friendly)
bookdatamaker export-dataset dataset.db -o output.csv -f csv

# JSON with metadata
bookdatamaker export-dataset dataset.db -o output.json -f json --include-metadata
```

### Compression Comparison

**For Parquet files:**

| Method | Speed | Size | Use Case |
|--------|-------|------|----------|
| `zstd` (default) | Fast | Small | Best balance, recommended |
| `snappy` | Fastest | Larger | Real-time processing |
| `gzip` | Medium | Smaller | Network transfer |
| `brotli` | Slowest | Smallest | Archival storage |
| `none` | Instant | Largest | Debug/testing only |

## Position Distribution

Control where threads start in the document using distribution percentages.

### How It Works

```
Document: 100 pages
Distribution: "10,10,20,30,20,10" (6 threads)

Thread 0: Start at 0%   → Page 1
Thread 1: Start at 10%  → Page 10
Thread 2: Start at 20%  → Page 20
Thread 3: Start at 50%  → Page 50
Thread 4: Start at 70%  → Page 70
Thread 5: Start at 80%  → Page 80
```

### Distribution Strategies

```bash
# Even distribution (4 threads)
--distribution "25,25,25,25"
# Start at: 0%, 25%, 50%, 75%

# Front-heavy (4 threads) - focus on beginning
--distribution "40,30,20,10"
# Start at: 0%, 40%, 70%, 90%

# Middle-heavy (5 threads) - focus on middle
--distribution "10,20,40,20,10"
# Start at: 0%, 10%, 30%, 70%, 90%

# Dense sampling (10 threads) - fine-grained coverage
--distribution "10,10,10,10,10,10,10,10,10,10"
```

### Thread Count Guidelines

- **Small documents** (<50 pages): 2-4 threads
- **Medium documents** (50-200 pages): 4-8 threads
- **Large documents** (>200 pages): 8-16 threads

---

## Performance Tuning

Optimize extraction and generation speeds based on your hardware and requirements.

### Stage 1: OCR Extraction

**API Mode (vLLM):**
```bash
# Increase concurrent requests (default: 4)
bookdatamaker extract book.pdf --api-concurrency 8

# Guidelines:
# - 2-4:  Small vLLM server (1-2 GPUs)
# - 4-8:  Medium server (2-4 GPUs)
# - 8-16: Large server (4+ GPUs)
# - Monitor server load and adjust accordingly
```

**Local Mode (Transformers):**
```bash
# Increase batch size (default: 8)
bookdatamaker extract book.pdf --mode local --batch-size 16

# Guidelines based on GPU VRAM:
# - 8GB VRAM:   batch-size 2-4
# - 16GB VRAM:  batch-size 4-8
# - 24GB VRAM:  batch-size 8-12
# - 40GB+ VRAM: batch-size 12-16
```

### Stage 2: Dataset Generation

**Thread Count:**
```bash
# More threads = faster generation (if LLM server can handle it)
bookdatamaker generate ./extracted \
  --distribution "10,10,10,10,10,10,10,10,10,10" \
  --threads 10

# Guidelines:
# - API mode: 4-16 threads (based on rate limits)
# - vLLM mode: 4-8 threads (based on GPU capacity)
# - Local mode: 2-4 threads (memory intensive)
```

**Message History Management:**
```bash
# Limit conversation history to prevent memory issues
bookdatamaker generate ./extracted \
  --max-messages 20 \
  -d dataset.db

# Default: 20 messages (system message + last 10 exchanges)
# Lower values = less memory, potentially less context
# Higher values = more memory, better context retention
```

**Duplicate Detection:**
- Automatically enabled with 95% similarity threshold
- Uses rapidfuzz for efficient fuzzy matching
- Prevents redundant Q&A pairs in the dataset

### Performance Tips

1. **Start Small**: Test with small concurrency/batch sizes first
2. **Monitor Resources**: Watch GPU memory, CPU usage, and network
3. **Balance Quality vs Speed**: Higher concurrency may reduce quality
4. **Network Bandwidth**: API mode performance depends on network speed
5. **vLLM Configuration**: Use tensor parallelism for multi-GPU setups

---

## Interactive Chat

Chat with an LLM that can access your document through MCP tools. Perfect for exploring documents interactively or testing Q&A generation.

### Start Chat Session

```bash
# Basic chat with GPT-4
bookdatamaker chat ./extracted

# With vLLM server
bookdatamaker chat ./extracted \
  --openai-api-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-4B-Thinking-2507

# With custom database
bookdatamaker chat ./extracted --db my_dataset.db
```

### Debug Mode

Set environment variable for verbose logging:

```bash
export LOG_LEVEL=DEBUG
bookdatamaker generate ./extracted -d dataset.db
```

---

## Development

### Project Structure

```
bookdatamaker/
├── src/bookdatamaker/
│   ├── cli.py                    # CLI interface
│   ├── ocr/
│   │   ├── extractor.py          # OCR extraction
│   │   └── document_parser.py    # Document parsing
│   ├── mcp/
│   │   └── server.py             # MCP server
│   ├── llm/
│   │   └── parallel_generator.py # Parallel generation
│   ├── dataset/
│   │   ├── builder.py            # Dataset building
│   │   └── dataset_manager.py    # SQLite management
│   └── utils/
│       ├── page_manager.py       # Page navigation
│       └── status.py             # Progress indicators
└── tests/                        # Test files
```

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/bookdatamaker.git
cd bookdatamaker

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black src/
ruff check src/

# Type checking
mypy src/
```

### Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ocr.py

# Run with coverage
pytest --cov=bookdatamaker tests/
```

---

## License

MIT License - see LICENSE file for details.
