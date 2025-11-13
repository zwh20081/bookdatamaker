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
- ⚡ **Resume Support**: Continue interrupted sessions

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
pip install -r requirements.txt && pip install -e .

# 2. Extract → Generate → Export
bookdatamaker extract book.pdf -o ./extracted
bookdatamaker generate ./extracted/combined.txt -d dataset.db --distribution "10,10,20,30,20,10"
bookdatamaker export-dataset dataset.db -o output.parquet
```

### Option 2: Self-Hosted Mode (Free, Private)

```bash
# 1. Install with local dependencies
pip install -r requirements.txt && pip install -e ".[local]"

# 2. Extract with local OCR
bookdatamaker extract book.pdf --mode local --batch-size 8 -o ./extracted

# 3. Generate with vLLM
bookdatamaker generate ./extracted/combined.txt \
  --mode vllm \
  --vllm-model-path meta-llama/Llama-3-8B-Instruct \
  --distribution "25,25,25,25" \
  -d dataset.db

# 4. Export
bookdatamaker export-dataset dataset.db -o output.parquet
```

## Installation

### Basic Installation

```bash
git clone https://github.com/yourusername/bookdatamaker.git
cd bookdatamaker
pip install -r requirements.txt
pip install -e .
```

### Optional: Local Inference Support

```bash
# For self-hosted OCR and LLM generation
pip install -e ".[local]"  # Installs transformers, torch, flash-attn
```

### System Requirements

**For API Mode:**
- Python 3.10+
- API keys (OpenAI, DeepSeek, etc.)

**For Local Mode:**
- Python 3.10+
- NVIDIA GPU with CUDA support
- 16GB+ VRAM recommended
- Linux or WSL2 (recommended)

---

## Extract Text (Stage 1)

Extract text from documents using DeepSeek OCR.

### Supported Formats

- **PDF**: Text extraction or OCR from rendered pages
- **EPUB**: E-book text extraction
- **Images**: JPG, PNG, BMP, TIFF, WebP

### API Mode

```bash
# Basic usage
bookdatamaker extract book.pdf -o ./extracted

# Custom API endpoint
bookdatamaker extract book.pdf \
  --deepseek-api-url https://custom-api.example.com/v1 \
  -o ./extracted
```

### Local Mode

Use local transformers model for OCR (no API calls):

```bash
# Basic usage
bookdatamaker extract book.pdf --mode local -o ./extracted

# With custom batch size (adjust based on GPU memory)
bookdatamaker extract book.pdf --mode local --batch-size 12 -o ./extracted

# Process directory of images
bookdatamaker extract ./images/ --mode local -o ./extracted
```

**Batch Size Guidelines:**
- **12-16**: GPUs with 24GB+ VRAM
- **8-12**: GPUs with 16GB+ VRAM (default: 8)
- **4-8**: GPUs with 8-12GB VRAM
- **1-4**: GPUs with <8GB VRAM

### Output Structure

```
./extracted/
├── page_001.txt
├── page_002.txt
├── ...
└── combined.txt    # All pages with [PAGE_XXX] markers
```

---

## Generate Dataset (Stage 2)

Generate Q&A datasets using parallel LLM threads.

### Basic Usage

```bash
# 6 threads (from distribution), 20 Q&A pairs per thread
bookdatamaker generate combined.txt \
  -d dataset.db \
  --distribution "10,10,20,30,20,10" \
  --datasets-per-thread 20
```

**Key Concept**: Thread count is determined by the number of comma-separated values in `--distribution`.

### API Mode Examples

```bash
# OpenAI/Azure
bookdatamaker generate combined.txt \
  -d dataset.db \
  --openai-api-url https://api.openai.com/v1 \
  --model gpt-4 \
  --distribution "10,10,20,30,20,10"

# Custom API endpoint
bookdatamaker generate combined.txt \
  --openai-api-url http://localhost:8000/v1 \
  --model your-model-name \
  --distribution "25,25,25,25"
```

### vLLM Direct Mode (Self-Hosted)

Use vLLM directly without API server:

```bash
# Single GPU
bookdatamaker generate combined.txt \
  --mode vllm \
  --vllm-model-path meta-llama/Llama-3-8B-Instruct \
  --distribution "25,25,25,25" \
  -d dataset.db

# Multi-GPU (4 GPUs, 6 threads)
bookdatamaker generate combined.txt \
  --mode vllm \
  --vllm-model-path meta-llama/Llama-3-70B-Instruct \
  --tensor-parallel-size 4 \
  --distribution "10,10,20,30,20,10" \
  -d dataset.db
```

**Benefits of vLLM Mode:**
- No API costs
- Full privacy (local processing)
- Optimized inference
- Thread-safe parallel processing
- Automatic batching

### Custom Prompts

Add specific instructions to guide LLM behavior:

```bash
# Language specification
bookdatamaker generate combined.txt \
  --custom-prompt "Generate all Q&A in Chinese with simplified characters"

# Format specification
bookdatamaker generate combined.txt \
  --custom-prompt "Questions should be multiple-choice with 4 options"

# Multiple requirements
bookdatamaker generate combined.txt \
  --custom-prompt "Requirements:
1. Generate questions in English
2. Focus on practical applications
3. Include code examples
4. Answer length: 50-150 words
5. Difficulty: intermediate"
```

---

## Export Dataset

Export from SQLite database to your preferred format:

```bash
# Parquet (recommended for data analysis)
bookdatamaker export-dataset dataset.db -o output.parquet

# JSON Lines (easy to stream)
bookdatamaker export-dataset dataset.db -o output.jsonl -f jsonl

# CSV (Excel-friendly)
bookdatamaker export-dataset dataset.db -o output.csv -f csv

# JSON with metadata
bookdatamaker export-dataset dataset.db -o output.json -f json --include-metadata
```

**Format Comparison:**

| Format | Best For | Size | Load Speed |
|--------|----------|------|------------|
| Parquet | Data analysis, ML | Smallest | Fastest |
| JSONL | Streaming, processing | Medium | Fast |
| CSV | Excel, spreadsheets | Largest | Medium |
| JSON | API responses | Large | Slow |

---

## Position Distribution

Control where threads start in the document using distribution percentages.

### How It Works

```
Document: 500 paragraphs
Distribution: "10,10,20,30,20,10" (6 threads)

Thread 0: Start at 0%   → Paragraph 1
Thread 1: Start at 10%  → Paragraph 50
Thread 2: Start at 20%  → Paragraph 100
Thread 3: Start at 50%  → Paragraph 250
Thread 4: Start at 70%  → Paragraph 350
Thread 5: Start at 80%  → Paragraph 400
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

- **Small documents** (<100 paragraphs): 2-4 threads
- **Medium documents** (100-500 paragraphs): 4-8 threads
- **Large documents** (>500 paragraphs): 8-16 threads

---

## Performance Tuning

### Extraction (Stage 1)

**Batch Size Optimization:**

```bash
# Maximum speed (24GB+ VRAM)
bookdatamaker extract book.pdf --mode local --batch-size 16

# Balanced (16GB VRAM)
bookdatamaker extract book.pdf --mode local --batch-size 8

# Conservative (<8GB VRAM)
bookdatamaker extract book.pdf --mode local --batch-size 4
```

### Generation (Stage 2)

**Optimal Configurations:**

```bash
# Maximum throughput (multi-GPU, 12 threads)
bookdatamaker generate text.txt --mode vllm \
  --vllm-model-path meta-llama/Llama-3-70B \
  --tensor-parallel-size 4 \
  --distribution "5,5,10,10,15,15,15,15,5,5,2,3" \
  --datasets-per-thread 50

# Balanced (single GPU, 6 threads)
bookdatamaker generate text.txt --mode vllm \
  --vllm-model-path meta-llama/Llama-3-8B \
  --distribution "10,10,20,30,20,10" \
  --datasets-per-thread 20

# Conservative (2 threads)
bookdatamaker generate text.txt --mode vllm \
  --vllm-model-path meta-llama/Llama-3-8B \
  --distribution "50,50" \
  --datasets-per-thread 10
```

---

## Interactive Chat

Chat with an LLM that can access your document through MCP tools. Perfect for exploring documents interactively or testing Q&A generation.

### Start Chat Session

```bash
# Basic chat with GPT-4
bookdatamaker chat combined.txt

# With vLLM server
bookdatamaker chat combined.txt \
  --openai-api-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-4B-Thinking-2507

# With custom database
bookdatamaker chat combined.txt --db my_dataset.db
```

### Example Interaction

```
📚 Document: combined.txt
📊 Paragraphs: 578
🤖 Model: gpt-4

You: What's in paragraph 100?
- `-f, --format`: Format: `jsonl`, `parquet`, `csv`, `json` (default: `parquet`)
- `--include-metadata`: Include timestamps

### Parameter Tables

#### extract Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | required | - | Input file or directory |
| `--output-dir` | optional | `extracted_text` | Output directory |
| `--mode` | optional | `api` | OCR mode: `api` or `local` |
| `--batch-size` | optional | `8` | Batch size for local mode |
| `--deepseek-api-key` | optional | env var | DeepSeek API key |
| `--deepseek-api-url` | optional | `https://api.deepseek.com/v1` | DeepSeek API URL |
| `--local-model-path` | optional | `deepseek-ai/DeepSeek-OCR` | Local model path |

#### generate Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text_file` | required | - | Combined text file |
| `--db` | optional | `dataset.db` | Database file path |
| `--mode` | optional | `api` | LLM mode: `api` or `vllm` |
| `--distribution` | optional | `10,10,20,30,20,10` | Position distribution (determines threads) |
| `--datasets-per-thread` | optional | `10` | Target Q&A pairs per thread |
| `--openai-api-key` | optional | env var | OpenAI API key |
| `--openai-api-url` | optional | `https://api.openai.com/v1` | API URL |
| `--model` | optional | `gpt-4` | Model name |
| `--vllm-model-path` | optional | - | vLLM model path |
| `--tensor-parallel-size` | optional | `1` | Number of GPUs |
| `--custom-prompt` | optional | - | Additional instructions |

---

## Troubleshooting

### Common Issues

**Problem: Threads not completing**
- Reduce `--datasets-per-thread`
- Check API rate limits
- Verify API keys
- Ensure document has enough content

**Problem: Out of memory (OCR)**
- Reduce `--batch-size`
- Use API mode instead of local

**Problem: Out of memory (Generation)**
- Reduce thread count (fewer distribution values)
- Use smaller model
- Reduce `--tensor-parallel-size`

**Problem: Low quality Q&A pairs**
- Adjust distribution to focus on content-rich sections
- Use higher-quality model (e.g., GPT-4)
- Add specific `--custom-prompt` instructions
- Check OCR quality

**Problem: SQLite errors**
- Ensure database path is writable
- Don't modify database during generation
- Delete and regenerate if corrupted

### Debug Mode

Set environment variable for verbose logging:

```bash
export LOG_LEVEL=DEBUG
bookdatamaker generate combined.txt -d dataset.db
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