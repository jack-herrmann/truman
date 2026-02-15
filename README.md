# Intuition — Agentic Personalities

Create **individual** and **consistent** artificial personalities. Not persona prompts or character sheets: deep, learned, psychologically grounded agents that behave consistently across contexts and distinctly from each other—so that interactions with AI can be genuinely meaningful.

## Why

The common denominator of deep personalities, stripped of ideological bias, is that they are **individual** and **consistent**. This project is an engine for such personalities: a high-dimensional latent space of Gaussians (mean = trait direction, variance = stability), a **prior** from classic literature, reinforcement learning in a **Truman Show**–style sandbox, and rewards for **consistency** and **individuality**. At test time we do latent-variable inference via Thompson sampling.

## Architecture

```
Novels (e.g. Gutenberg; scraping via Bright Data when needed)
    │
    ▼
CharacterExtractor (LLM: e.g. NVIDIA NeMo) ──► CharacterDataset ──► Embeddings
                                                                         │
                                                                         ▼
                                                                   PersonalityVAE
                                                                         │
                                                          ┌──────────────┴──────────────┐
                                                          ▼                              ▼
                                                   z (latent mean)                 σ (per-dim stability)
                                                          │                              │
                                                          └──────────────┬──────────────┘
                                                                         │
                                                                         ▼
                                                          PersonalityDecoder (LLM)
                                                                         │
                                                                         ▼
                                                                   PersonalityKernel
                                                                         │
                                                          ┌──────────────┼──────────────┐
                                                          ▼              ▼              ▼
                                                   PromptBuilder   StateManager   EpisodicMemory
                                                          │              │              │
                                                          └──────────────┼──────────────┘
                                                                         ▼
                                                                   PersonalityAgent
                                                                         │
                                                                         ▼
                                                              TrumanWorld (sandbox)
                                                    LLM-generated non-deterministic social situations
                                                                         │
                                                          ┌──────────────┼──────────────┐
                                                          ▼              ▼              ▼
                                                  CoherenceReward  IndividualityReward  Probes
                                                          │              │              │
                                                          └──────────────┼──────────────┘
                                                                         ▼
                                                                   KernelOptimizer
                                                              (evolutionary search)
```

## Core Concepts

**Personality as latent Gaussians**: Each personality is a point in a learned latent space: **mean** (z) encodes trait direction, **variance** (σ) encodes per-dimension stability. Some traits are bedrock (low σ); others are fault lines—strong one day, softer the next.

**Prior from literature**: We need to “show” the model what a deep personality looks like. Think Dostoevsky, Kafka, Austen: characters so deep you can’t really picture them until you’ve lived with them through the story. We obtain text from online sources (e.g. Project Gutenberg; **Bright Data** for harder scraping), extract character profiles with an LLM (e.g. **NVIDIA NeMo**), and use these as the prior for the VAE.

**Truman Show environment**: We use an LLM to generate and adapt a **sandbox** of non-deterministic social situations. Our entities act in this world—our own little Truman Show. No right answers; response depends on who you are.

**Rewards**: We avoid ideological bias by keeping the reward simple: **consistency** (same z → stable behavioral signature; we measure via probes and narrative consistency) and **individuality** (different z → meaningfully different behavior; we measure via discriminability and population spread in feature space). We also add random noise to avoid unwanted convergence. At test time: latent-variable inference via Thompson sampling.

## Quick Start

```bash
pip install -r requirements.txt
```

### Option A: Free API (recommended to start)

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey) (1,500 requests/day) or [Groq Console](https://console.groq.com):

```bash
export GEMINI_API_KEY="your-key-here"   # or GROQ_API_KEY
python3 demo.py
```

### Option B: Paid API or NVIDIA NeMo

```bash
export ANTHROPIC_API_KEY="your-key-here"   # or OPENAI_API_KEY
# Or use NVIDIA NeMo / NIM: set NIM_PROXY_BASE_URL and provider: nemo in config.yaml
# Edit config.yaml → llm.provider
python3 demo.py
```

### Option C: Use as a library

```python
import asyncio
from intuition import create_personality, create_agent

async def main():
    kernel = await create_personality()
    agent = await create_agent(kernel)
    print(f"Meet: {kernel.name}")
    response = await agent.respond("What matters most to you in life?")
    print(response)

asyncio.run(main())
```

### LLM Providers

| Provider | Env Variable | Notes |
|----------|-------------|--------|
| `gemini` | `GEMINI_API_KEY` | Free tier (default) |
| `groq` | `GROQ_API_KEY` | Free tier, OpenAI-compatible |
| `nemo` | `NIM_PROXY_BASE_URL` | NVIDIA NeMo / NIM (self-hosted or NGC) |
| `anthropic` | `ANTHROPIC_API_KEY` | Paid |
| `openai` | `OPENAI_API_KEY` | Paid |

Set `llm.provider` in `config.yaml` (and optional `llm.base_url` for NeMo).

## Seed Data

The repo ships with **pre-extracted character profiles** from classic literature (Raskolnikov, Elizabeth Bennet, Heathcliff, etc.), pre-computed embeddings, and an optional pre-trained VAE checkpoint so you can create personalities without running the full pipeline.

To regenerate from the shipped profiles:

```bash
python3 scripts/generate_seed_data.py
```

## Full Training Pipeline

```bash
# 1. Download novels (e.g. Project Gutenberg; optional: USE_BRIGHTDATA=1 for Bright Data)
python3 scripts/download_corpus.py

# 2. Extract character profiles (requires LLM API key; e.g. NeMo or Gemini)
python3 scripts/extract_characters.py

# 3. Train the personality VAE
python3 scripts/train_vae.py

# 4. Run evolutionary personality training (Truman Show + consistency/individuality rewards)
python3 scripts/train_personalities.py

# 5. Evaluate trained personalities
python3 scripts/evaluate.py
```

## Demos

- **`draw_yourself.py`** — **Draw yourself**: a projective drawing demo. Each personality draws a self-portrait from its kernel (no LLM at draw time). Inspired by the projective drawing tasks used in personality development research—here we let our agentic personalities “draw themselves.” Run with `--save out.png`; use `--kernel path/to/kernel.json` to skip the LLM. See the docstring in `draw_yourself.py` for full steps. (`person_in_rain.py` is a launcher that runs this.)
- **`demo.py`** — Create 3 personalities, run them through 3 situations, compare behavior side-by-side.
- **`demos/conversation.py`** — Interactive chat with one personality. Load a saved kernel or create one.
- **`demos/interpolate.py`** — Interpolate between two saved personalities; optionally decode the midpoint with the LLM.

```bash
# Draw yourself (needs API key to create 4, or use --kernel to skip LLM)
python3 draw_yourself.py --save my_drawing.png
python3 draw_yourself.py --kernel data/checkpoints/test_kernel.json --no-window --save out.png

# Main demo
python3 demo.py

# Chat with a saved personality
python3 demos/conversation.py --kernel data/checkpoints/final_top1.json

# Interpolate two kernels
python3 demos/interpolate.py kernel_a.json kernel_b.json --decode-mid
```

## Testing

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -v
```

Tests cover core models (kernel, traces, memory), latent space (VAE, PersonalitySpace, interpolation), agent (StateManager, PromptBuilder), rewards, API, and demos.

## Project Structure

```
intuition/
├── core/           # PersonalityKernel, BehavioralTrace, EpisodicMemory
├── llm/            # LLMClient (NeMo, OpenAI, Anthropic, Gemini, Groq), EmbeddingClient, TemplateEngine
├── corpus/         # GutenbergCorpus (optional Bright Data fetcher), CharacterExtractor, CharacterDataset
├── latent/         # PersonalityVAE, PersonalitySpace, PersonalityDecoder
├── agent/          # PersonalityAgent, PromptBuilder, StateManager
├── environment/    # SituationBank, WorldNarrator, Curriculum, TrumanWorld
├── training/       # CoherenceReward, IndividualityReward, KernelOptimizer, TrainingLoop
├── evaluation/     # PersonalityProbe, ConsistencyEvaluator, IndividualityEvaluator
└── api.py          # create_personality, create_agent
prompts/            # Jinja2 templates
scripts/            # download_corpus, extract_characters, train_vae, train_personalities, evaluate
demos/              # conversation.py, interpolate.py
tests/
config.yaml         # llm (provider, base_url), corpus (use_brightdata), paths
```

## Configuration

- **`config.yaml`** — LLM provider (gemini, groq, nemo, anthropic, openai), NeMo base URL, embeddings, latent dimension, training weights, corpus paths. For NeMo set `llm.provider: nemo` and `NIM_PROXY_BASE_URL` or `llm.base_url`. For Bright Data scraping set `corpus.use_brightdata: true` or `USE_BRIGHTDATA=1` (requires `BRIGHTDATA_API_TOKEN`).

## Beyond the demo

The same architecture becomes a platform: accurately simulated populations for enterprise testing, highly personalized agents (e.g. meaningful connection in care settings), and other applications where individual, consistent personalities matter.
