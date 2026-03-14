# Paper Data — Cross-Substrate Coupling Preprint

Replication data for: *Jointly Tracing Coupling Mode Transitions in Human-AI Interaction: Semantic and Somatic Signals*

## Session Data

Three sessions analysed in detail, organised by subdirectory:

| Session | Directory | Model | Biosignal | Paper Sections |
|---------|-----------|-------|-----------|----------------|
| **Grunch** | `grunch-session/` | GPT-OSS-20B | 1,360 cardiac samples | §4.1, §4.2 |
| **ZoryaGPT** | `zoryagpt-session/` | Custom GPT (OpenAI) | 734 cardiac samples | §4.3 |
| **Echo** | `echo-session/` | Claude Sonnet 4 | None (semantic-only) | §4.4 |

### File Types

- `*.json` — Semantic Climate session exports (conversation turns, metrics history, embedding trajectories)
- `*.md` — Session transcripts (ZoryaGPT, conducted via ChatGPT interface)
- `biosignal_*.jsonl` — Raw cardiac data from Polar H10 via EarthianBioSense (one JSON object per sample: timestamp, HR, RR intervals)
- `biosignal_*_processed.jsonl` — Processed biosignal with HRV metrics, entrainment, coherence scores

### Session Corpus Manifest

`session_corpus_manifest.csv` — Index of all 37 recordings (35 valid + 2 empty/aborted) with metadata: session ID, model, provider, turn count, duration, biosignal availability, embedding model. The full corpus is available from the corresponding author on request.

## Reproducing Figures

**Prerequisites:** Python 3.9+, numpy, matplotlib, scipy, scikit-learn

```bash
# Figures 1 and 2 (from EarthianLabs root)
python publications/papers/figures/generate_figures.py

# Figure 3 — turn-level coupling analysis
python semantic-climate-phase-space/tools/turn_level_coupling.py
```

Figure 1: Grunch dual time-series (HR + α + ΔH, denial phase annotated)
Figure 2: Cross-session comparison (three panels)
Figure 3: Turn-level coupling analysis (§4.6 — null result at turn boundaries, phase-level signal)

## Metric Implementation

The semantic coupling metrics (Δκ, α, ΔH) are implemented in `semantic-climate-phase-space/src/core_metrics.py`, extending Morgoulis (2025) with corrections documented in §3.2 of the paper. Key differences from the original implementation:

- **Δκ (Semantic Curvature):** Local Frenet-Serret curvature, not chord deviation
- **α (Fractal Similarity):** DFA on semantic velocity (inter-turn cosine distances), not embedding norms
- **ΔH (Entropy Shift):** JS divergence on shared clustering, not independent clustering

Biosignal processing (HRV, entrainment, coherence) is implemented in `Earthian-BioSense/src/processing/`.
