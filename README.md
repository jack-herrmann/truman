# Intuition — AI Personality Engine

Create truly **individual** and **coherent** artificial personalities. Not persona prompts. Not character sheets. Deep, learned, psychologically grounded personalities that behave consistently across contexts and distinctly from each other.

## Architecture

```
Novels (Gutenberg)
    │
    ▼
CharacterExtractor ──► CharacterDataset ──► Embeddings
                                                │
                                                ▼
                                          PersonalityVAE
                                                │
                                     ┌──────────┴──────────┐
                                     ▼                      ▼
                              z (latent vector)      σ (stability)
                                     │                      │
                                     └──────────┬──────────┘
                                                │
                                                ▼
                                     PersonalityDecoder (LLM)
                                                │
                                                ▼
                                       PersonalityKernel
                                                │
                              ┌─────────────────┼─────────────────┐
                              ▼                 ▼                 ▼
                       PromptBuilder      StateManager     EpisodicMemory
                              │                 │                 │
                              └─────────────────┼─────────────────┘
                                                ▼
                                       PersonalityAgent
                                                │
                                                ▼
                                          TrumanWorld
                                       (situations → traces)
                                                │
                              ┌─────────────────┼─────────────────┐
                              ▼                 ▼                 ▼
                      CoherenceReward   IndividualityReward   Probes
                              │                 │                 │
                              └─────────────────┼─────────────────┘
                                                ▼
                                        KernelOptimizer
                                     (evolutionary search)
```

## Quick Start

```bash
pip install -r requirements.txt
```

### Option A: Free API (recommended to start)

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey) (1,500 requests/day) or [Groq Console](https://console.groq.com) (free tier):

```bash
# Pick one:
export GEMINI_API_KEY="your-key-here"   # Google Gemini (default)
export GROQ_API_KEY="your-key-here"     # Groq (llama-3.3-70b)

# Run the demo — creates 3 personalities and compares them
python3 demo.py
```

### Option B: Paid API

```bash
export ANTHROPIC_API_KEY="your-key-here"  # or OPENAI_API_KEY
# Edit config.yaml → provider: anthropic (or openai)
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

| Provider | Env Variable | Default Model | Cost |
|----------|-------------|---------------|------|
| `gemini` | `GEMINI_API_KEY` | `gemini-2.0-flash` | Free (1,500 req/day) |
| `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | Free tier |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` | Paid |
| `openai` | `OPENAI_API_KEY` | `gpt-4o` | Paid |

Set the provider in `config.yaml` or pass it directly: `LLMClient(provider="gemini")`.

## Seed Data

The repo ships with **27 pre-extracted character profiles** from classic literature (Raskolnikov, Elizabeth Bennet, Heathcliff, etc.) plus pre-computed embeddings and a pre-trained VAE checkpoint. This means you can go straight to personality creation without running the full training pipeline.

To regenerate the seed embeddings and VAE from the shipped profiles:

```bash
python3 scripts/generate_seed_data.py
```

## Full Training Pipeline

```bash
# 1. Download novels from Project Gutenberg
python3 scripts/download_corpus.py

# 2. Extract character profiles (requires LLM API key)
python3 scripts/extract_characters.py

# 3. Train the personality VAE
python3 scripts/train_vae.py

# 4. Run evolutionary personality training
python3 scripts/train_personalities.py

# 5. Evaluate trained personalities
python3 scripts/evaluate.py
```

## Core Concepts

**Personality as latent vector**: Each personality is a point `z` in a learned latent space, with `σ` encoding per-dimension stability. High σ = fault lines. Low σ = bedrock traits.

**PersonalityKernel**: A rich Pydantic model capturing values, cognitive style, emotional baseline, social style, perceptual filters, internal contradictions (fault lines), and stress responses. Not a list of traits — a coherent psychology.

**Novels as training data**: Classic literature provides character profiles with the depth and coherence that personality taxonomies lack. The VAE learns a smooth space from these characters.

**Truman Show environment**: An LLM-generated world of ambiguous, identity-revealing situations. No right answers — response depends on who you are. Includes boredom, moral dilemmas, social conflict, escalation, and contradiction triggers.

**Coherence reward**: Same `z` → consistent behavioral signature across contexts. Measured via embedding similarity, probe stability, and LLM-as-judge narrative consistency.

**Individuality reward**: Different `z` → meaningfully different behavior. Measured via discriminability (can a classifier tell agents apart?), pairwise behavioral distance, and qualitative judgment.

## Project Structure

```
intuition/
├── core/           # PersonalityKernel, BehavioralTrace, EpisodicMemory
├── llm/            # LLMClient, EmbeddingClient, TemplateEngine
├── corpus/         # GutenbergCorpus, CharacterExtractor, CharacterDataset
├── latent/         # PersonalityVAE, PersonalitySpace, PersonalityDecoder
├── agent/          # PersonalityAgent, PromptBuilder, StateManager
├── environment/    # SituationBank, WorldNarrator, Curriculum, TrumanWorld
├── training/       # CoherenceReward, IndividualityReward, KernelOptimizer, TrainingLoop
├── evaluation/     # PersonalityProbe, ConsistencyEvaluator, IndividualityEvaluator
└── api.py          # Public API: create_personality, create_agent
prompts/            # Jinja2 templates for all LLM interactions
scripts/            # Pipeline entry points
config.yaml         # Centralized configuration
```
