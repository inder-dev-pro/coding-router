<div align="center">

# coding-router

**A lightweight LLM router that classifies coding requests and selects the best-fit model using semantic similarity and cost-aware scoring.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-coding--router-orange.svg)](https://pypi.org/project/coding-router/)

<br>
<img src="assets/output.webp" width="720" alt="Demo">

</div>

---

Based on the [LLMRouter paper](paper/2608.06867v1.pdf) (Feng et al., 2026):

```
π* = argmax_π  E[ perf(y | q) − λ · c(τ) ]

→  reward_m = α · perf_normalized(m) − β · cost_normalized(m)
```

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Platform Notes](#platform-notes)
  - [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [LLM Gateway](#llm-gateway)
- [Dashboard](#dashboard)
- [Routing Modes](#routing-modes)
- [Classifier Backends](#classifier-backends)
- [Model Catalog](#model-catalog)
- [Evaluation & Benchmarks](#evaluation--benchmarks)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [Development](#development)
- [License](#license)

---

## Features

- **Local classifier** — runs the [ModelGate](https://huggingface.co/AaryanK/ModelGate) router classifier locally via a quantised GGUF model (auto-downloaded, ~600 MB one-time). No external server needed.
- **Semantic routing** — embeds your query and every model description with [`Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), then ranks by cosine similarity + cost penalty.
- **Cost-aware selection** — five built-in routing modes from quality-only to cost-dominant, plus arbitrary `(α, β)` overrides.
- **Unified Model Calling** — uses [LiteLLM](https://github.com/BerriAI/litellm) for robust, multi-provider model invocation with a single standard interface.
- **LLM Gateway** — an OpenAI/Anthropic-compatible HTTP proxy that transparently routes requests to the best model.
- **Live Dashboard** — a web UI that visualises routing events, model distribution, cost tracking, and latency metrics from a local SQLite database.
- **Cross-platform** — Metal acceleration on macOS, CPU/CUDA on Windows and Linux.
- **Project-local config** — catalog lives in `./router/coding_llm.json`, safe to commit to git.
- **Extensible catalog** — add your own local or cloud models via `coding-router add-model`.

---

## Architecture

The system follows a multi-stage pipeline architecture where each stage is a pure function operating on a shared state dictionary:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        coding-router Pipeline                        │
│                                                                      │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────────┐  │
│  │   User      │   │  Embedding   │   │     Classification       │  │
│  │   Query     │──▶│  (Qwen 0.6B) │──▶│  (ModelGate GGUF / vLLM) │  │
│  └─────────────┘   │              │   │                          │  │
│                     │  Encodes     │   │  Categorises into:       │  │
│                     │  query to    │   │  • code_generation       │  │
│                     │  768-dim     │   │  • bug_fixing            │  │
│                     │  vector      │   │  • performance_optim.    │  │
│                     └──────────────┘   │  • api_help              │  │
│                                        │  • programming           │  │
│                                        └──────────────────────────┘  │
│                                                  │                    │
│                                                  ▼                    │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │                      Model Ranking                            │   │
│  │                                                               │   │
│  │  For each model m in catalog:                                 │   │
│  │    1. cosine_sim = dot(query_embedding, model_embedding)      │   │
│  │    2. capability_fit = cosine_sim + category_boost(m)         │   │
│  │    3. perf_norm = min_max_normalize(capability_fit)           │   │
│  │    4. cost_norm = min_max_normalize(estimated_cost(m))        │   │
│  │    5. reward = α · perf_norm − β · cost_norm                 │   │
│  │                                                               │   │
│  │  Sort by (reward, perf_norm, -cost) descending                │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │                   Selection & Gating                          │   │
│  │                                                               │   │
│  │  1. Choose group: standard vs advanced                        │   │
│  │     (complex queries → advanced; heuristic + regex patterns)  │   │
│  │  2. Quality-gap cost-aware decision rule:                     │   │
│  │     If a cheaper model is within threshold of the best,       │   │
│  │     pick the cheaper one                                      │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│                    ┌──────────────────┐                               │
│                    │  Selected Model  │                               │
│                    │  (+ invocation)  │                               │
│                    └──────────────────┘                               │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Overview

```
coding_router/
├── router.py           # Core pipeline: CodingRouter class + scoring (Eq. 1)
├── classifier.py       # LocalClassifier (GGUF) + VLLMClassifier (remote)
├── embeddings.py       # Qwen3-Embedding-0.6B encoder + on-disk index
├── catalog.py          # ModelProfile dataclass + JSON catalog loader
├── config.py           # Config resolution chain + env var management
├── quantize.py         # GGUF download, caching, platform-specific kwargs
├── invocation.py       # Provider adapters (OpenAI, Anthropic, Google)
├── cli_ui.py           # ANSI terminal UI (banner, spinner, progress bar)
├── __main__.py         # CLI entry point + all subcommands
├── gateway/            # FastAPI-based OpenAI/Anthropic-compatible proxy
│   ├── server.py       #   App factory + lifespan + auth middleware
│   ├── routing.py      #   Bridge between gateway requests and router
│   ├── auth.py         #   API key generation and validation
│   ├── errors.py       #   Error hierarchy + exception handlers
│   ├── models.py       #   Pydantic models for gateway req/res
│   ├── streaming.py    #   SSE streaming utilities
│   └── providers/
│       └── base.py     #   LiteLLM-based multi-provider adapter
│   └── protocols/
│       ├── openai_chat.py        # POST /v1/chat/completions
│       ├── openai_responses.py   # POST /v1/responses
│       └── anthropic_messages.py # POST /v1/messages
└── dashboard/          # Web-based analytics dashboard
    ├── server.py       #   Lightweight HTTP server (daemon thread)
    ├── tracker.py      #   SQLite event recorder + analytics queries
    └── static/         #   Single-page HTML/JS/CSS dashboard
```

---

## Getting Started

### Installation

```bash
pip install coding-router
```

For gateway support (OpenAI/Anthropic-compatible proxy):

```bash
pip install coding-router[gateway]
```

For development:

```bash
pip install coding-router[dev]
```

### Platform Notes

| Platform | What happens |
| --- | --- |
| **macOS (Apple Silicon)** | `llama-cpp-python` auto-uses Metal for GPU acceleration |
| **macOS (Intel)** | CPU-only inference, works fine for the classifier |
| **Windows** | CPU by default; install `llama-cpp-python` with CUDA support for GPU |
| **Linux** | GPU offloading attempted by default; falls back to CPU |

### Quick Start

#### 1. Initialise the project

```bash
cd your-project/
coding-router init
```

This creates:

- `router/coding_llm.json` — model catalog (all models included)
- `.env.example` — lists the API key env vars you need to set

#### 2. Edit the catalog

Open `router/coding_llm.json` and **delete any models you don't have access to**. Every model present in the file is available for routing.

#### 3. Set API keys

```bash
# Copy the example and fill in your keys
cp .env.example .env

# Or export directly:
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
```

> **Note:** API keys are never stored in the catalog. They're always read from environment variables at runtime. `router/coding_llm.json` is safe to commit to git.

#### 4. Validate your setup

```bash
coding-router validate
```

This checks:

- ✓ JSON structure is valid
- ✓ All required fields are present
- ✓ Cloud models have their API keys set
- ✓ Local model endpoints are reachable

#### 5. Route a query

```bash
# Route without calling a provider (dry run):
coding-router --route-only "Fix an intermittent race condition in our Python worker"

# Route and invoke the selected model:
coding-router "Design a caching layer for this API"
```

---

## CLI Reference

| Command | What it does |
| --- | --- |
| `coding-router init` | Create `router/` dir with catalog + `.env.example` |
| `coding-router validate` | Check catalog, API keys, and endpoints |
| `coding-router models` | List all models in the catalog with pricing |
| `coding-router add-model` | Add a custom model to the user catalog |
| `coding-router "query"` | Route (and optionally invoke) a query |
| `coding-router --route-only "query"` | Select a model without invoking |
| `coding-router serve` | Start the OpenAI/Anthropic-compatible gateway |
| `coding-router dashboard` | Start the analytics dashboard server |
| `coding-router doctor` | Diagnose gateway configuration and readiness |

### Route Options

```bash
coding-router "your query" \
  --mode mixed \                  # Routing policy (see modes below)
  --alpha 0.7 \                   # Override performance weight α
  --beta 0.3 \                    # Override cost weight β
  --route-only \                  # Don't invoke — just select
  --force-advanced \              # Force advanced model tier
  --show-ranking \                # Emit full ranking as JSON
  --pool all \                    # "all" or "user" (custom models only)
  --classifier-backend local \    # "llama-server", "local", or "vllm"
  --max-output-tokens 2048        # Output token budget
```

---

## Python API

### Basic usage

```python
from coding_router import CodingRouter, RouterConfig

router = CodingRouter()
result = router.route("Fix this race condition", route_only=True)

print(result["selected_model"]["catalog_key"])  # e.g. "claude-sonnet-5"
print(result["classifier_category"])            # e.g. "bug_fixing"
print(result["selected_model"]["reward"])        # cost-adjusted score
```

### Custom configuration

```python
from pathlib import Path
from coding_router import CodingRouter, RouterConfig

router = CodingRouter(RouterConfig(
    catalog_path=Path("router/coding_llm.json"),
    routing_mode="cost_efficient",       # favour cheaper models
    classifier_backend="local",          # in-process GGUF classifier
    target_max_tokens=2048,
    quality_gap_threshold=0.05,          # cost-aware decision rule threshold
))
```

### Route method signature

```python
result = router.route(
    query,                              # The coding request
    route_only=False,                   # If True, don't invoke the model
    force_advanced=False,               # Force advanced model tier
    force_no_gating=False,              # Skip standard/advanced gating
    routing_mode="mixed",               # Override config's routing mode
    alpha=None,                         # Override α weight
    beta=None,                          # Override β weight
    quality_gap_threshold=None,         # Override gap threshold
    estimated_output_tokens=None,       # Override output token estimate
)
```

### Result structure

The `route()` method returns a `RouterState` dictionary containing:

| Key | Type | Description |
| --- | --- | --- |
| `selected_model` | `dict` | Selected model with all metadata and scores |
| `classifier_category` | `str` | One of the 5 task categories |
| `classifier_raw_response` | `str` | Raw classifier output |
| `selection_group` | `str` | `"standard_model_group"` or `"advanced_model_group"` |
| `selection_reason` | `str` | Why this group was chosen |
| `ranked_models` | `list[dict]` | All models ranked by reward score |
| `model_response` | `str` | The LLM response (if `route_only=False`) |
| `invocation_error` | `str` | Error message if invocation failed |

---

## LLM Gateway

The gateway exposes an OpenAI/Anthropic-compatible HTTP API that transparently routes requests to the best model via the router. It uses [LiteLLM](https://github.com/BerriAI/litellm) for unified multi-provider support.

### Start the gateway

```bash
# Basic usage
coding-router serve

# With custom settings
coding-router serve \
  --host 0.0.0.0 \
  --port 8080 \
  --mode cost_sensitive \
  --no-auth \
  --log-level debug
```

### Pre-flight check

```bash
coding-router doctor
```

This validates the catalog, API keys, router initialisation, and required dependencies (`fastapi`, `uvicorn`, `litellm`).

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | OpenAI Chat Completions (streaming + non-streaming) |
| `POST` | `/v1/responses` | OpenAI Responses API |
| `POST` | `/v1/messages` | Anthropic Messages API (streaming + non-streaming) |
| `GET` | `/v1/models` | List all available models |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive Swagger documentation |

### Use with any OpenAI SDK client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="<your-gateway-key>",  # printed on startup
)

# Automatic routing — the gateway picks the best model
response = client.chat.completions.create(
    model="coding-router",  # virtual model name
    messages=[{"role": "user", "content": "Optimize this database query"}],
)

# Or target a specific model by catalog key
response = client.chat.completions.create(
    model="claude-sonnet-5",  # bypass routing
    messages=[{"role": "user", "content": "Write a REST API"}],
)
```

### Use with Anthropic SDK

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8080/v1",
    api_key="<your-gateway-key>",
)

message = client.messages.create(
    model="coding-router",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Fix this memory leak"}],
)
```

### Authentication

The gateway generates a random API key on first run (stored in `~/.config/coding-router/gateway.key`). Pass it via `Authorization: Bearer <key>` or `x-api-key: <key>` header. Disable with `--no-auth`.

---

## Dashboard

The dashboard provides a web-based analytics interface for monitoring routing decisions, model usage distribution, cost tracking, and latency metrics.

```bash
# Start standalone
coding-router dashboard --port 3000

# It also starts automatically when the router is used
```

All routing events are recorded to a local SQLite database at `~/.cache/coding-router/dashboard.db`. The dashboard serves a single-page app with a JSON API.

### Dashboard API Endpoints

| Path | Description |
| --- | --- |
| `/api/summary` | Overall stats (total requests, unique models, avg latency, total cost) |
| `/api/events` | Paginated event list with full routing details |
| `/api/model-distribution` | Model selection frequency breakdown |
| `/api/category-distribution` | Task category distribution |
| `/api/latency-timeline` | Latency over time |
| `/api/cost-timeline` | Cumulative cost tracking |

---

## Routing Modes

The paper defines five operating points on the performance–cost trade-off curve:

| Mode | α (quality) | β (cost) | Use it when |
| --- | ---: | ---: | --- |
| `skill_based` | 1.0 | 0.0 | Best semantic match regardless of price |
| `quality_leaning` | 0.8 | 0.2 | Slight cost awareness |
| `mixed` (default) | 0.6 | 0.4 | Balanced quality/cost decision |
| `cost_sensitive` | 0.4 | 0.6 | Cost matters more than quality |
| `cost_efficient` | 0.2 | 0.8 | Cost should dominate |

Override with `--alpha` and `--beta` for arbitrary sweep points:

```bash
coding-router --alpha 0.9 --beta 0.1 "your query"
```

### Quality-Gap Decision Rule

After ranking, the router applies a cost-aware decision rule: if a cheaper model is within `quality_gap_threshold` (default: 0.05) of the top-ranked model's performance score, it selects the cheaper alternative. This prevents overspending when multiple models are similarly suited.

---

## Classifier Backends

The classifier categorises queries into one of five task routes: `code_generation`, `bug_fixing`, `performance_optimization`, `api_help`, and `programming`.

### Local GGUF (default, recommended)

Loads the quantised GGUF model in-process via `llama-cpp-python`. The model is auto-downloaded from Hugging Face on first use (~600 MB, one-time).

```bash
coding-router --classifier-backend local "your query"
```

### llama-server

Auto-starts `llama serve -hf AaryanK/ModelGate:Q8_0` as a background process. The model stays loaded across calls for fast subsequent requests.

Requires `llama.cpp` installed:

```bash
brew install llama.cpp  # macOS
```

### Remote server (vLLM / Ollama)

Connect to any OpenAI-compatible server:

```bash
coding-router --classifier-backend vllm \
  --classifier-base-url http://localhost:8000/v1 \
  "your query"
```

---

## Model Catalog

The catalog (`router/coding_llm.json`) defines every model available for routing. Each entry includes:

```json
{
  "cloud_coding_models": {
    "claude-sonnet-5": {
      "size": "Undisclosed (mid-tier)",
      "feature": "Claude Sonnet 5 is Anthropic's production workhorse...",
      "input_price": 2.00,
      "output_price": 10.00,
      "model": "claude-sonnet-5",
      "service": "Anthropic",
      "api_endpoint": "https://api.anthropic.com/v1",
      "used_in": ["Claude Code", "Cursor", "GitHub Copilot"]
    }
  }
}
```

### Adding custom models

```bash
coding-router add-model \
  --key my-local-coder \
  --model codellama:34b \
  --endpoint http://localhost:11434/v1 \
  --feature "Local CodeLlama for fast Python generation and debugging" \
  --service "Self-hosted (Ollama / vLLM)" \
  --size "34B" \
  --input-price 0 \
  --output-price 0 \
  --tier standard
```

User models are stored separately in `router/user_models.json` and merged with the curated catalog at runtime. Use `--pool user` to route only among your custom models.

### Model tiers

Models are split into two tiers:

- **Standard** — general-purpose coding models for everyday tasks
- **Advanced** — flagship models for complex, long-horizon work (architecture, multi-file refactors, security audits)

The router automatically selects the advanced tier when it detects complex queries via keyword patterns or when the query exceeds 2,000 characters.

---

## Evaluation & Benchmarks

The project includes a comprehensive evaluation framework built on the [RouterBench](https://arxiv.org/abs/2403.12031) methodology, using the MBPP (Mostly Basic Python Programming) dataset.

### Run the benchmark

```bash
python -m evaluation.run_mbpp_benchmark
python -m evaluation.run_mbpp_benchmark --limit 50  # quick test
```

### Evaluation components

| Module | Purpose |
| --- | --- |
| `evaluation/routerbench/loader.py` | Load and preprocess MBPP benchmark data |
| `evaluation/routerbench/runner.py` | Execute the router against benchmark prompts |
| `evaluation/routerbench/oracle.py` | Compute oracle (best-possible) model selections |
| `evaluation/routerbench/metrics.py` | Model-level and summary statistics |
| `evaluation/routerbench/baselines.py` | Baseline comparisons (cheapest, random, semantic-only) |
| `evaluation/routerbench/plots.py` | Generate visualisation charts |
| `evaluation/routerbench/report.py` | Export results as CSV, JSON, and formatted reports |

---

## Configuration Reference

### Config resolution

The router looks for the model catalog in this order:

1. Explicit `--catalog` path passed on the CLI
2. `LLMROUTER_CONFIG` environment variable
3. `./router/coding_llm.json` (project-local)
4. Error telling you to run `coding-router init`

### Environment variables

| Variable | Purpose |
| --- | --- |
| `LLMROUTER_CONFIG` | Override catalog path (default: `./router/coding_llm.json`) |
| `CODING_ROUTER_CACHE_DIR` | Override model cache directory (default: `~/.cache/coding-router/`) |
| `CODING_ROUTER_AUTH` | Enable/disable gateway auth (`true`/`false`) |
| `CODING_ROUTER_MODEL_NAME` | Override the virtual model name (default: `coding-router`) |

### Provider API keys

| Variable | Provider |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Google |
| `XAI_API_KEY` | xAI |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `DASHSCOPE_API_KEY` | Alibaba/Qwen |
| `MISTRAL_API_KEY` | Mistral AI |
| `ZAI_API_KEY` | Z.ai (Zhipu) |
| `MOONSHOT_API_KEY` | Moonshot AI |
| `MINIMAX_API_KEY` | MiniMax |

Per-model API key overrides are also supported via `MODEL_ROUTER_<KEY>_API_KEY` (e.g., `MODEL_ROUTER_CLAUDE_SONNET_5_API_KEY`).

---

## Project Structure

```
router/
├── coding_router/              # Core Python package
│   ├── __init__.py             # Public API exports
│   ├── __main__.py             # CLI entry point (7 subcommands)
│   ├── router.py               # CodingRouter class + pipeline stages
│   ├── classifier.py           # GGUF + vLLM classifiers
│   ├── embeddings.py           # Qwen encoder + on-disk index
│   ├── catalog.py              # Model catalog loader + user models
│   ├── config.py               # Config resolution + env var management
│   ├── quantize.py             # GGUF download + platform detection
│   ├── invocation.py           # Direct provider adapters
│   ├── cli_ui.py               # ANSI terminal UI
│   ├── gateway/                # OpenAI/Anthropic-compatible HTTP proxy
│   │   ├── server.py           #   FastAPI app + middleware
│   │   ├── routing.py          #   Gateway ↔ router bridge
│   │   ├── providers/base.py   #   LiteLLM multi-provider adapter
│   │   └── protocols/          #   OpenAI Chat, Responses, Anthropic
│   ├── dashboard/              # Web analytics dashboard
│   │   ├── server.py           #   HTTP server (daemon thread)
│   │   ├── tracker.py          #   SQLite event recorder
│   │   └── static/             #   SPA front-end
│   └── data/                   # Bundled catalog template
├── dashboard/                  # Standalone dashboard web app (Vite + React)
├── data/                       # Benchmark data + model embeddings
├── evaluation/                 # RouterBench evaluation framework
│   ├── run_mbpp_benchmark.py   #   Main benchmark entry point
│   └── routerbench/            #   Loader, runner, metrics, plots, reports
├── tests/                      # Unit + integration tests
│   ├── test_router_basics.py   #   Core routing logic tests
│   └── test_gateway.py         #   Gateway API tests
├── paper/                      # Research paper + implementation guides
├── arch/                       # Architecture diagrams
├── pyproject.toml              # Package config + dependencies
├── requirements.txt            # Minimal dependency list
└── LICENSE                     # MIT License
```

---

## Development

### Setup

```bash
git clone https://github.com/indersharma/coding-router.git
cd coding-router
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,gateway]"
```

### Run tests

```bash
# Core routing tests
pytest tests/test_router_basics.py -v

# Gateway tests
pytest tests/test_gateway.py -v

# All tests
pytest tests/ -v
```

### Build the embedding index

```bash
coding-router --build-index
```

This re-embeds all catalog model descriptions using Qwen3-Embedding-0.6B and persists the vectors to `router/model_embeddings.npz`. The index rebuilds automatically when the catalog changes.

### Lint

```bash
ruff check coding_router/
```

---

## License

[MIT](LICENSE) © Inder Sharma
