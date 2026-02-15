# The Story — Agentic Personalities

We'll all spend more and more time interacting with AI. Why not make those interactions genuinely **meaningful**?

This project creates **individual** and **consistent** artificial personalities. Not persona prompts or character sheets: deep, learned, psychologically grounded agents that behave consistently across contexts and distinctly from each other.

> **Heads up — this project makes heavy use of LLM API calls** (creating priors, generating sandbox environments, decoding personalities). Free-tier quotas (Gemini, Groq) **will** be exhausted during normal use. Before you set up any API keys, see what the system actually produces by opening the pre-saved demo below.

---

## See It First (no install, no API key, nothing)

Remember when your mum used to hang your drawing on the fridge? Psychologists use projective drawing tasks to generate hypotheses about personality development. It's simple: *draw yourself.*

We can compare what happens when children draw themselves, when current LLMs draw themselves, and when our agentic personalities draw themselves.

Open **[`demos/draw_yourself_presaved.html`](demos/draw_yourself_presaved.html)** in your browser.

You'll see four personalities — each drawing a self-portrait in real time. The Stoic draws a plain, centered figure with no embellishment. The Anxious one produces a small, hunched form surrounded by scribbled-out false starts. The Romantic fills the canvas with flowing lines, flowers, and stars. The Pragmatist sketches a neatly dressed figure with glasses and polished shoes.

**This is not a mockup.** These drawings are a materialization of real output from the personality pipeline. Each of the four agents was created through the full process — latent vector sampling, LLM-based personality generation, kernel-to-drawing-spec mapping — and the resulting drawing commands were transcribed into the HTML file so you can see exactly what the system produces without spending a single API call. The only manual additions were to make the HTML look a bit nicer. The personalities, their traits, and the drawing decisions are all real; the HTML is just a recording.

Pretty cool! But why? Beyond the proof-of-concept, this becomes a platform: accurately simulated populations for enterprise testing, highly personalized agents — for example, in old age, meaningful connections are one of the most important health indicators — and any application where individual, consistent personalities matter.

---

## Try It Yourself (30 seconds, no API key)

```bash
pip install -r requirements.txt
python3 draw_yourself.py --kernel data/checkpoints/test_kernel.json --no-window --save out.png
```

Open `out.png` — you should see a self-portrait drawn entirely from a saved personality kernel (no LLM needed at draw time). The figure's posture, mood, position, and whether it carries an umbrella all come from the personality's latent vector, stress profile, and emotional baseline.

> If you get `No module named 'pydantic'` or similar, make sure you ran `pip install -r requirements.txt` first (use a virtual environment if your system Python is managed).

---

## Create Your Own Personalities (needs a free API key)

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey) or [Groq Console](https://console.groq.com), then:

```bash
export GEMINI_API_KEY="your-key-here"    # or GROQ_API_KEY for Groq

# Draw 4 new personalities (creates them via LLM, then draws)
python3 draw_yourself.py --save my_drawing.png

# Compare 3 personalities side-by-side across moral dilemmas
python3 demo.py

# Chat with the shipped test personality
python3 demos/conversation.py --kernel data/checkpoints/test_kernel.json
```

**Free-tier rate limits:** Gemini free tier has a daily request quota. If you hit `Rate limit exceeded`, wait a minute and retry, or switch to Groq (set `GROQ_API_KEY` and change `llm.provider` to `groq` in `config.yaml`).

---

## Why

The common denominator of deep personalities, stripped of ideological bias, is that they are **individual** and **consistent**. This project is an engine for such personalities: a high-dimensional latent space of Gaussians (mean = trait direction, variance = stability), a **prior** from classic literature, reinforcement learning in a **Truman Show**-style sandbox, and rewards for **consistency** and **individuality**. At test time we do latent-variable inference via Thompson sampling.

## Architecture

```
Novels (e.g. Gutenberg; scraping via Bright Data when needed)
    |
    v
CharacterExtractor (LLM: e.g. NVIDIA Nemotron) --> CharacterDataset --> Embeddings
                                                                        |
                                                                        v
                                                                  PersonalityVAE
                                                                        |
                                                         +--------------+--------------+
                                                         v                              v
                                                  z (latent mean)                 s (per-dim stability)
                                                         |                              |
                                                         +--------------+--------------+
                                                                        |
                                                                        v
                                                         PersonalityDecoder (LLM)
                                                                        |
                                                                        v
                                                                  PersonalityKernel
                                                                        |
                                                         +--------------+--------------+
                                                         v              v              v
                                                  PromptBuilder   StateManager   EpisodicMemory
                                                         |              |              |
                                                         +--------------+--------------+
                                                                        |
                                                                        v
                                                                  PersonalityAgent
                                                                        |
                                                                        v
                                                             TrumanWorld (sandbox)
                                                   LLM-generated non-deterministic social situations
                                                                        |
                                                         +--------------+--------------+
                                                         v              v              v
                                                 CoherenceReward  IndividualityReward  Probes
                                                         |              |              |
                                                         +--------------+--------------+
                                                                        |
                                                                        v
                                                                  KernelOptimizer
                                                             (evolutionary search)
```

## Core Concepts

**Personality as latent Gaussians.** Each personality is a point in a learned latent space: **mean** (z) encodes trait direction, **variance** (sigma) encodes per-dimension stability. Some traits are bedrock (low sigma); others are fault lines — strong one day, softer the next.

**Prior from literature.** We need to "show" the model what a deep personality looks like. Think Dostoevsky, Kafka, Austen: characters so deep you can't really picture them until you've lived with them through the story. We scrape these stories from online sources (e.g. Project Gutenberg; **Bright Data** made this challenge super smooth), then extract individual character profiles with an LLM (e.g. **NVIDIA Nemotron**). These then form our priors.

**Truman Show environment.** RL requires an environment and actions. We use an LLM to generate and adapt a **sandbox** of non-deterministic social situations — our very own little **Truman Show**. Our entities act in this world; no right answers, response depends on who you are.

**Rewards.** We avoid ideological bias by keeping the reward simple: **consistency** (same z -> stable behavioral signature; we measure via probes and narrative consistency) and **individuality** (different z -> meaningfully different behavior; we measure via discriminability and population spread in feature space). We also add random noise to avoid unwanted convergence. At test time: latent-variable inference via Thompson sampling.

---

## Demos

### demos/draw_yourself_presaved.html — Pre-saved demo (no install, no API key)

Open in any browser. Four personality archetypes draw self-portraits in real time. This is a transcript of real output from the pipeline — the personalities, traits, and drawing decisions were generated by the system and recorded as a standalone HTML file.

### draw_yourself.py — Self-portrait (works without API key)

Each personality draws a self-portrait from its kernel. The figure's posture, mood, position, and umbrella all come from the kernel's latent vector, stress profile, and emotional baseline. No LLM at draw time.

```bash
# Use the shipped test kernel (no API key needed)
python3 draw_yourself.py --kernel data/checkpoints/test_kernel.json --save out.png

# Create 4 new personalities and draw them (needs API key)
python3 draw_yourself.py --save my_drawing.png

# Headless (server, no display) — always saves a file
python3 draw_yourself.py --kernel data/checkpoints/test_kernel.json --no-window --save out.png
```

### demo.py — Side-by-side comparison (needs API key)

Creates 3 personalities, runs them through identical moral dilemmas, prints a side-by-side comparison of their perceptions, emotions, and actions.

```bash
python3 demo.py
```

### demos/conversation.py — Interactive chat (needs API key)

Chat in the terminal with a personality. Load a saved kernel or create a new one.

```bash
# Chat with the shipped test personality
python3 demos/conversation.py --kernel data/checkpoints/test_kernel.json

# Create a new personality and chat
python3 demos/conversation.py

# Create one "like" a literary character (needs VAE + embeddings)
python3 demos/conversation.py --like "Elizabeth Bennet"
```

### demos/interpolate.py — Latent interpolation (needs 2 saved kernels)

Interpolate between two saved PersonalityKernel files in latent space. Requires two kernel JSON files (e.g. from training or saved from `create_personality`).

```bash
python3 demos/interpolate.py kernel_a.json kernel_b.json
python3 demos/interpolate.py --decode-mid kernel_a.json kernel_b.json   # decode midpoint via LLM
```

> **Note:** This demo requires two kernel files. You get these from the training pipeline or by saving personalities from `demo.py` / the Python API.

---

## Use as a Library

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

---

## LLM Providers

| Provider | Env Variable | Notes |
|----------|-------------|--------|
| `gemini` | `GEMINI_API_KEY` | Free tier (default). Daily quota — if you hit rate limits, wait or switch to Groq. |
| `groq` | `GROQ_API_KEY` | Free tier, OpenAI-compatible. Good alternative if Gemini quota is exhausted. |
| `nemo` | `NIM_PROXY_BASE_URL` | NVIDIA Nemotron / NIM (self-hosted or NGC). Set `llm.provider: nemo` in config.yaml. |
| `anthropic` | `ANTHROPIC_API_KEY` | Paid. |
| `openai` | `OPENAI_API_KEY` | Paid. |

Set `llm.provider` in `config.yaml` and export the matching env variable.

---

## Seed Data

The repo ships with **34 pre-extracted character profiles** from classic literature (Raskolnikov, Elizabeth Bennet, Heathcliff, etc.) in `data/characters/`, plus a test personality kernel in `data/checkpoints/test_kernel.json`.

To regenerate embeddings and the VAE from these profiles:

```bash
python3 scripts/generate_seed_data.py
```

## Full Training Pipeline

```bash
# 1. Download novels (Project Gutenberg)
python3 scripts/download_corpus.py

# 2. Extract character profiles (requires LLM API key)
python3 scripts/extract_characters.py

# 3. Train the personality VAE
python3 scripts/train_vae.py

# 4. Evolutionary personality training (Truman Show + consistency/individuality rewards)
python3 scripts/train_personalities.py

# 5. Evaluate trained personalities
python3 scripts/evaluate.py
```

After training, saved kernels appear in `data/checkpoints/` (e.g. `final_top1.json`). You can use these with any demo's `--kernel` flag.

---

## Bright Data Integration

Novel downloads go through a pluggable `PageFetcher` abstraction. By default, plain HTTP (`httpx`) is used. For sites that need proxies or anti-bot bypassing, enable **Bright Data**:

```bash
# 1. Install the SDK (included in requirements.txt)
pip install brightdata-sdk

# 2. Set your API token (get one at https://brightdata.com → Dashboard → API tokens)
export BRIGHTDATA_API_TOKEN="your-token"

# 3. Enable Bright Data — pick any one of these:
python3 scripts/download_corpus.py --brightdata          # CLI flag
USE_BRIGHTDATA=1 python3 scripts/download_corpus.py      # env var
# or set corpus.use_brightdata: true in config.yaml      # config file
```

Under the hood, `BrightDataFetcher` uses the official [Bright Data Python SDK](https://pypi.org/project/brightdata-sdk/) (`client.scrape.generic.url_async`) for each novel URL. The fetcher plugs into `GutenbergCorpus` — same interface, different transport:

```python
from intuition.corpus import GutenbergCorpus, BrightDataFetcher

corpus = GutenbergCorpus(fetcher=BrightDataFetcher())
await corpus.download_all()
```

---

## Configuration

**`config.yaml`** controls LLM provider, model, embeddings, latent dimensions, training hyperparameters, and file paths.

Key settings:
- `llm.provider` — which LLM to use (gemini, groq, nemo, anthropic, openai)
- `llm.model` — model name (e.g. `gemini-2.0-flash`, `llama-3.3-70b-versatile`)
- `llm.base_url` — for NeMo/NIM: the proxy endpoint
- `corpus.use_brightdata` — enable Bright Data scraping (`true`, env `USE_BRIGHTDATA=1`, or `--brightdata`)
- `BRIGHTDATA_API_TOKEN` — API token for Bright Data (env var, auto-read by SDK)

## Project Structure

```
intuition/
  core/           PersonalityKernel, BehavioralTrace, EpisodicMemory
  llm/            LLMClient (NeMo, OpenAI, Anthropic, Gemini, Groq), EmbeddingClient
  corpus/         GutenbergCorpus (optional Bright Data fetcher), CharacterExtractor
  latent/         PersonalityVAE, PersonalitySpace, PersonalityDecoder
  agent/          PersonalityAgent, PromptBuilder, StateManager
  environment/    SituationBank, WorldNarrator, Curriculum, TrumanWorld
  training/       CoherenceReward, IndividualityReward, KernelOptimizer
  evaluation/     PersonalityProbe, ConsistencyEvaluator, IndividualityEvaluator
  api.py          create_personality, create_agent
prompts/          Jinja2 templates
scripts/          download_corpus, extract_characters, train_vae, train_personalities, evaluate
demos/            draw_yourself_presaved.html, conversation.py, interpolate.py
data/
  characters/     34 pre-extracted character profiles (JSON)
  checkpoints/    test_kernel.json (shipped), trained models (after pipeline)
config.yaml
```

## Testing

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -v
```

---

We all know the saying: you are a mixture of the five people you spend the most time with. If an artificial entity is one of those five, it'd be foolish not to have it be a genuinely meaningful, positive presence.

**We've learned so much building this and we hope to have contributed our little part. May we all flourish.**
