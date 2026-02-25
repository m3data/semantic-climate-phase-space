# Spawn Guide — RAA-SC-001

## Prerequisites

- Hence v0.6.x installed (`hence --version`)
- This directory has a git repo (`git init` if needed)
- Initial commit includes plan.spl + CLAUDE.md + empty findings dirs

## Known Issues

- `hence watch --spawn` does NOT work — use manual spawns below
- Parallel spawns cause plan.spl merge conflicts — copy findings from worktrees manually
- Agent scope can drift — check output files match expected task scope

## Phase A — Independent Audit (parallel)

Run all at once. Each spawns in its own worktree:

```bash
hence spawn plan.spl --agent claude --task a1-dfa-alpha-fractal-similarity
hence spawn plan.spl --agent claude --task a2-delta-kappa-semantic-curvature
hence spawn plan.spl --agent claude --task a3-delta-h-entropy-shift-jsd
hence spawn plan.spl --agent claude --task a4-psi-vector-assembly
hence spawn plan.spl --agent claude --task a5-coupling-mode-classification
```

**After all complete:** Copy findings from worktrees to main repo:

```bash
# Find worktree paths (they'll be under .claude/worktrees/ or similar)
git worktree list

# Copy each finding
cp <worktree-a1>/findings-a/01-dfa-alpha-fractal-similarity.md findings-a/
cp <worktree-a2>/findings-a/02-delta-kappa-semantic-curvature.md findings-a/
cp <worktree-a3>/findings-a/03-delta-h-entropy-shift-jsd.md findings-a/
cp <worktree-a4>/findings-a/04-psi-vector-assembly.md findings-a/
cp <worktree-a5>/findings-a/05-coupling-mode-classification.md findings-a/
```

Then mark completions:

```bash
hence complete plan.spl a1-dfa-alpha-fractal-similarity --agent claude
hence complete plan.spl a2-delta-kappa-semantic-curvature --agent claude
hence complete plan.spl a3-delta-h-entropy-shift-jsd --agent claude
hence complete plan.spl a4-psi-vector-assembly --agent claude
hence complete plan.spl a5-coupling-mode-classification --agent claude
```

## Phase A Synthesis

```bash
hence spawn plan.spl --agent claude --task a-synthesis
```

Copy output:

```bash
cp <worktree-synthesis>/findings-a/SYNTHESIS-A.md findings-a/
hence complete plan.spl a-synthesis --agent claude
```

## Phase B — Adversarial Counter-Audit (parallel)

```bash
hence spawn plan.spl --agent claude --task b1-counter-dfa-alpha-fractal-similarity
hence spawn plan.spl --agent claude --task b2-counter-delta-kappa-semantic-curvature
hence spawn plan.spl --agent claude --task b3-counter-delta-h-entropy-shift-jsd
hence spawn plan.spl --agent claude --task b4-counter-psi-vector-assembly
hence spawn plan.spl --agent claude --task b5-counter-coupling-mode-classification
```

**After all complete:** Copy findings from worktrees:

```bash
cp <worktree-b1>/findings-b/01-counter-dfa-alpha-fractal-similarity.md findings-b/
cp <worktree-b2>/findings-b/02-counter-delta-kappa-semantic-curvature.md findings-b/
cp <worktree-b3>/findings-b/03-counter-delta-h-entropy-shift-jsd.md findings-b/
cp <worktree-b4>/findings-b/04-counter-psi-vector-assembly.md findings-b/
cp <worktree-b5>/findings-b/05-counter-coupling-mode-classification.md findings-b/
```

Then mark completions:

```bash
hence complete plan.spl b1-counter-dfa-alpha-fractal-similarity --agent claude
hence complete plan.spl b2-counter-delta-kappa-semantic-curvature --agent claude
hence complete plan.spl b3-counter-delta-h-entropy-shift-jsd --agent claude
hence complete plan.spl b4-counter-psi-vector-assembly --agent claude
hence complete plan.spl b5-counter-coupling-mode-classification --agent claude
```

## Evaluator — Convergence Assessment

```bash
hence spawn plan.spl --agent claude --task convergence-eval
```

Copy output:

```bash
cp <worktree-eval>/CONVERGENCE-REPORT.md .
hence complete plan.spl convergence-eval --agent claude
```

## Cleanup

```bash
# List worktrees
git worktree list

# Remove all audit worktrees
git worktree list --porcelain | grep "^worktree " | cut -d' ' -f2 | xargs -I{} git worktree remove {}
```

## Expected Timeline

- Phase A: ~5 parallel agents, ~5-10 min each
- Phase A Synthesis: ~3 min
- Phase B: ~5 parallel agents, ~5-10 min each
- Evaluator: ~5 min
- Total: ~25-35 min wall clock (with parallelism)

## Preprint Impact Assessment

The evaluator's CONVERGENCE-REPORT.md should specifically assess whether any finding affects:
- The α = 0.77–1.27 lock-in claim (load-bearing for Grunch finding)
- The ΔH = 0.228 fragmentation spike
- The Ψ vector composition described in preprint Table 1
- The basin classification labels used in the preprint narrative
