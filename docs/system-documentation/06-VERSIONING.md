# Schema Versioning

This document describes the module-level versioning system for Semantic Climate Phase Space, enabling research reproducibility by tracking exactly which analysis implementations produced each session export.

## Overview

Session exports now include version information for each analysis module. This allows:

1. **Reproducibility** — Know exactly which algorithms produced results
2. **Comparability** — Determine if sessions are comparable or need re-analysis
3. **Migration** — Identify pre-versioning sessions and infer likely versions from dates
4. **Forward compatibility** — Load older sessions while understanding their provenance

## Module Versions

Each module that affects analysis output has its own semantic version:

| Module | Current | Description |
|--------|---------|-------------|
| `core_metrics` | 1.1.0 | Morgoulis metrics (Δκ, α, ΔH) |
| `extensions` | 1.1.0 | Ψ vector, trajectory dynamics |
| `basin_detection` | 2.0.0 | Attractor basin classification |
| `export_schema` | 2.0.0 | Session export format |
| `affective` | 1.0.0 | VADER sentiment analysis |
| `biosignal` | 1.0.0 | EBS integration |

## Version History

### Core Metrics (`core_metrics`)

**1.1.0 (2025-12-08)** — All three metrics fixed
- Δκ: Frenet-Serret local curvature (was chord deviation)
- α: DFA on semantic velocity (was embedding norms)
- ΔH: Shared clustering + JS divergence (was independent clustering)

**1.0.0 (2025-12-06)** — Initial Morgoulis implementation

### Extensions (`extensions`)

**1.1.0 (2025-12-11)** — Basin detection v2 integration
- 9 canonical basins with dialogue context
- Trajectory metrics inform classification
- Coherence pattern integration

**1.0.0 (2025-12-06)** — Initial extensions
- Ψ vector (semantic, temporal, affective, biosignal)
- TrajectoryBuffer for dynamics
- Basic attractor basin detection (7 basins)

### Basin Detection (`basin_detection`)

**2.0.0 (2025-12-11)** — Refined v2 with dialogue context
- Added trajectory metrics (velocity, acceleration, curvature)
- Added coherence pattern detection
- Expanded to 9 basins with nuanced thresholds

**1.0.0 (2025-12-06)** — Initial basin detection (7 basins)

### Export Schema (`export_schema`)

**2.0.0 (2025-12-11)** — Added version tracking
- `metadata.schema_version` for format version
- `metadata.versions` block with all module versions

**1.0.0 (pre-2025-12-11)** — Original export format
- No version tracking

## Session Export Format

Session exports now include:

```json
{
  "metadata": {
    "session_id": "uuid",
    "created_at": "ISO timestamp",
    "exported_at": "ISO timestamp",
    "schema_version": "2.0.0",
    "versions": {
      "core_metrics": "1.1.0",
      "extensions": "1.1.0",
      "basin_detection": "2.0.0",
      "export_schema": "2.0.0",
      "affective": "1.0.0",
      "biosignal": "1.0.0"
    },
    // ... other metadata
  }
}
```

## API Usage

### Get Current Versions

```python
from src import get_current_versions, get_versions_dict

# As dataclass
versions = get_current_versions()
print(versions.core_metrics)  # "1.1.0"

# As dict (for embedding in exports)
versions_dict = get_versions_dict()
```

### Check Session Compatibility

```python
from src import check_compatibility, needs_reanalysis

# Load session metadata
session_versions = session_data["metadata"].get("versions", {})

# Check each module
compat = check_compatibility(session_versions)
for module, info in compat.items():
    print(f"{module}: {info['note']}")

# Quick check if re-analysis needed
if needs_reanalysis(session_versions):
    print("Session should be re-analyzed with current versions")
```

### Infer Versions for Pre-Versioning Sessions

```python
from src import infer_versions_from_date

# For sessions without version info
export_date = session_data["metadata"]["exported_at"]
inferred = infer_versions_from_date(export_date)
# Returns best-guess versions based on known release dates
```

## Migration Strategy

### Identifying Pre-Versioning Sessions

Sessions exported before 2025-12-11 lack version info. These can be identified by:

```python
if "versions" not in session_data["metadata"]:
    # Pre-versioning session
    inferred = infer_versions_from_date(session_data["metadata"]["exported_at"])
```

### Re-Analysis Workflow

For research requiring comparable results:

1. Identify sessions needing re-analysis:
   ```python
   from src import needs_reanalysis

   sessions_to_reanalyze = [
       s for s in sessions
       if needs_reanalysis(s["metadata"].get("versions", {}))
   ]
   ```

2. Re-run analysis using `tools/reanalyze_session.py`

3. Archive original exports in `research/archive-sessions/`

4. Store re-analyzed exports with current versions

## Versioning Guidelines

When modifying analysis code:

1. **Increment version** in `src/schema.py` for the affected module
2. **Add changelog entry** documenting the change
3. **Update this document** if adding new modules or major changes
4. **Consider backwards compatibility** — can old sessions still be loaded?

### Version Number Semantics

- **Major** (X.0.0): Breaking changes that invalidate previous results
- **Minor** (0.X.0): New features or significant refinements
- **Patch** (0.0.X): Bug fixes that don't change interpretation

## Related Documentation

- `A2-METRIC-FIX-VALIDATION.md` — Details of the core metrics fixes in v1.1.0
- `A1-METRICS-MATHEMATICAL-REVIEW.md` — Mathematical analysis behind fixes
- `02-EXTENSIONS.md` — Ψ vector and basin detection details
