# CLAUDE.md — Recursive Adversarial Audit RAA-SC-001

You are performing a recursive adversarial audit.

## What You're Doing

- **Phase A agents** audit code claims independently
- **Phase B agents** adversarially review Phase A findings
- **Evaluator** measures convergence between passes

## Target

Semantic Climate Phase Space metric computation pipeline at `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/`.

This repository implements the Semantic Climate Phase Space model, extending Morgoulis (2025) 4D Semantic Coupling Framework. It provides three core metrics (Δκ, α, ΔH), a composite Ψ vector (semantic, temporal, affective, biosignal substrates), and attractor basin detection for classifying dialogue coupling dynamics. The cross-substrate coupling preprint (`/Users/m3untold/Code/EarthianLabs/publications/papers/cross-substrate-coupling-preprint.md`) relies on these implementations — particularly the DFA/α computation, which is load-bearing for the Grunch lock-in finding (α = 0.77–1.27 during epistemic rupture).

**Use absolute paths** to read files. The target may be outside this repo but is accessible on the filesystem.

## Key Files

- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/src/core_metrics.py` — Δκ (semantic_curvature_enhanced), α (fractal_similarity_robust), ΔH (entropy_shift_comprehensive), and calculate_all_metrics orchestrator
- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/src/substrates.py` — Ψ substrate computations (semantic, temporal, affective, biosignal)
- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/src/basins.py` — BasinDetector, BasinHistory, hysteresis, soft membership, 10 basin taxonomy
- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/src/trajectory.py` — TrajectoryBuffer, geometry computations
- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/src/analyzer.py` — SemanticClimateAnalyzer orchestrator
- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/src/extensions.py` — Legacy monolith (deprecated, but may still be imported)
- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/src/integrity.py` — IntegrityAnalyzer, TransformationDetector
- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/src/api.py` — Function-based API
- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/src/schema.py` — Version tracking
- `/Users/m3untold/Code/EarthianLabs/semantic-climate-phase-space/tests/` — Test suite (113+ tests across multiple files)

## Preprint Reference

The cross-substrate coupling preprint is at `/Users/m3untold/Code/EarthianLabs/publications/papers/cross-substrate-coupling-preprint.md`. Section 3.2 describes the metric implementations. Section 4 reports empirical findings. The audit should assess whether the code supports the preprint's claims.

## Output Format

### Phase A (Audit) — write to `findings-a/`

```
## Claim
[What the code claims to do]

## Files Examined
[Absolute paths and line ranges]

## Evidence
[What was found — describe at design-pattern level]

## Finding
**Verdict: CONFIRMED | PARTIAL | NOT CONFIRMED | CONCERN**
[One-paragraph explanation]

## Notes
[Edge cases, caveats, potential weaknesses]
```

### Phase B (Counter-Audit) — write to `findings-b/`

```
## Phase A Finding Under Review
[Reference to the Phase A finding]

## Phase A Verdict
[What Phase A concluded]

## Counter-Evidence
[What Phase B found that Phase A missed, got wrong, or understated]

## Revised Assessment
**Verdict: AGREE | UPGRADE | DOWNGRADE | CONTEST**
- AGREE: Phase A finding is accurate
- UPGRADE: Phase A was too conservative — the code is better than reported
- DOWNGRADE: Phase A was too generous — the code is worse than reported
- CONTEST: Phase A finding is materially wrong

## Convergence Notes
[Where this finding agrees vs. disagrees with Phase A]
```

### Evaluator — write to `CONVERGENCE-REPORT.md`

Per-finding convergence scores (0.0–1.0), overall confidence, structurally
unresolvable findings, and prioritised remediation actions.

## Constraints

- **Read-only** on the target codebase — do NOT modify target files
- Describe logic at **design-pattern level** — include function names and line numbers but avoid excessive code copying
- Phase B agents: use **INDEPENDENT analysis** — re-read the source code, don't just critique Phase A's prose
- **Stay within your task scope** — do not write findings for other tasks or phases

## Tone

Technical, honest, constructive. The goal is to find where claims exceed implementation, not to score points.
