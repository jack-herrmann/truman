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

# Create a personality (requires ANTHROPIC_API_KEY)
python3 -c "
import asyncio
from intuition import create_personality, create_agent

async def main():
    kernel = await create_personality()
    agent = await create_agent(kernel)
    print(f'Meet: {kernel.name}')
    response = await agent.respond('What matters most to you in life?')
    print(response)

asyncio.run(main())
"
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
