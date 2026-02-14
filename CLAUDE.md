# Truman — AI Personality Architecture

## Project Vision

Create AI agents with genuinely **individual and coherent personalities** — not surface-level
persona prompts, but deep personality structures that are consistent without rigidity, emerge
from underlying values/experiences, shape what agents notice (not just how they respond),
and meaningfully differentiate one agent from another.

Named after *The Truman Show*: agents live in LLM-generated worlds that stress-test and
develop their personalities.

## Core Architecture

### Personality Kernel — Latent Vector `z ~ N(μ, σ)`
- Uninterpretable latent space (like StyleGAN), not predefined trait dimensions
- `μ` = personality mean (core stable identity)
- `σ` = per-dimension variance encoding trait stability/resolution
- Policy conditioned on `z`: same situation + different `z` = different behavior

### State vs. Trait Dynamics
- `z_trait` (stable kernel) + `z_state` (dynamic emotional/cognitive state)
- Personality shapes *how* states are experienced, not whether they occur
- Personality determines the manifold of states visited

### Adaptation and Drift
- Slow `μ` updates gated by experience intensity → character arcs
- Most interactions leave `z` unchanged; extreme/repeated experiences shift baseline

### Multi-modal Distributions
- Some personalities are bimodal (public/private self), not just high variance
- Contextual switching function determines active mode
- Models personality compartmentalization as coherent, not inconsistent

### Episodic Memory
- Self-narrative reinforces personality: "I'm the kind of person who did X"
- Feeds back to state (short-term) and kernel (long-term drift)
- Stabilizing force beyond the reward signal

## Training Pipeline

### Phase 1: Novel-Bootstrapped Latent Space
- Extract behavioral traces from literary characters (decisions, dialogue, internal monologue,
  attention patterns, stress responses)
- Pre-train autoencoder on character profiles → learned personality space
- New agents sample from this novel-informed prior

### Phase 2: RL in "Truman Show" Environments
- LLM-generated text-based life simulator (DM creates personality-revealing situations)
- Narrative action space: say, do, notice, feel, prioritize
- Ambiguous social/personal situations (no obvious right answer)
- Stress-testing: contradictory pressures, escalation, boredom/emptiness

### RL Reward (dual objective in tension)
- **Coherence**: discriminator clusters behavioral traces by `z` origin (stylometric consistency)
- **Individuality**: maximize `I(z; behavior)` (InfoGAN-inspired mutual information)
- **Narrative consistency**: behavior makes sense given agent's own history

## Measurement

### Consistency
- Personality test batteries (MBTI x100, Big Five, attachment, moral foundations, etc.)
- Adversarial probes: reframed questions, temptation scenarios

### Individuality
- Discriminability (classifier accuracy identifying agent from responses)
- Behavioral distance (pairwise distances — want spread, not clustering)
- Surprise relative to population (unpredictability vs. average agent)

## Demo Concepts
1. **Prediction Game** (strongest): audience sees 10 responses, predicts new dilemma → proves personality is *knowable*
2. **Cocktail Party**: 6-8 agents in group conversation with emergent roles + crisis injection
3. **Long Interview**: talk to agent A, switch to B (visceral difference), bring back A (recognition)

## Tech Stack (anticipated from .gitignore)
- Python (ML/RL training)
- Node.js (environment/demo layer)
- Possible compiled components (C/C++)
