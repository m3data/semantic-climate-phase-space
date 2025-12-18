"""
Semantic Climate Phase Space - Schema Versioning

Tracks module-level versions for research reproducibility.
Each component that affects analysis output has its own version.

When session exports include this version info, you can determine
exactly which implementations produced the results.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import json

# =============================================================================
# MODULE VERSIONS
# =============================================================================

# Core metrics (Morgoulis base + fixes)
CORE_METRICS_VERSION = "1.1.0"
# Changelog:
# 1.1.0 (2025-12-08) - All three metrics fixed
#   - Δκ: Frenet-Serret local curvature (was chord deviation)
#   - α: DFA on semantic velocity (was embedding norms)
#   - ΔH: Shared clustering + JS divergence (was independent clustering)
# 1.0.0 (2025-12-06) - Initial Morgoulis implementation
#   - Original algorithms from 4d-semantic-coupling

# Extensions (Ψ vector, attractors, trajectory)
EXTENSIONS_VERSION = "1.1.0"
# Changelog:
# 1.1.0 (2025-12-11) - Basin detection v2 refined
#   - 9 canonical basins with dialogue context awareness
#   - Trajectory metrics inform basin classification
#   - Coherence pattern integration
# 1.0.0 (2025-12-06) - Initial extensions
#   - Ψ vector (semantic, temporal, affective, biosignal)
#   - TrajectoryBuffer for dynamics
#   - Basic attractor basin detection (7 basins)

# Basin detection (subset of extensions, tracked separately for granularity)
BASIN_DETECTION_VERSION = "2.0.0"
# Changelog:
# 2.0.0 (2025-12-11) - Refined v2 with dialogue context
#   - Added trajectory metrics (velocity, acceleration, curvature)
#   - Added coherence pattern detection
#   - Expanded to 9 basins with nuanced thresholds
# 1.0.0 (2025-12-06) - Initial basin detection
#   - 7 canonical basins
#   - Threshold-based classification

# Export schema (the format of session exports)
EXPORT_SCHEMA_VERSION = "2.0.0"
# Changelog:
# 2.0.0 (2025-12-11) - Added version tracking
#   - metadata.versions block with all module versions
#   - metadata.schema_version for export format itself
# 1.0.0 (pre-2025-12-11) - Original export format
#   - Basic metadata (session_id, timestamps, model names)
#   - No version tracking for analysis components

# Affective substrate (VADER + contextual)
AFFECTIVE_VERSION = "1.0.0"
# Changelog:
# 1.0.0 (2025-12-06) - Initial implementation
#   - VADER sentiment analysis
#   - Contextual intensity modulation

# Biosignal substrate (EBS integration)
BIOSIGNAL_VERSION = "1.0.0"
# Changelog:
# 1.0.0 (2025-12-06) - Initial implementation
#   - HR normalization (40-180 bpm range)
#   - Basic entrainment/coherence pass-through


# =============================================================================
# VERSION AGGREGATION
# =============================================================================

@dataclass
class AnalysisVersions:
    """Complete version snapshot for an analysis run."""
    core_metrics: str
    extensions: str
    basin_detection: str
    export_schema: str
    affective: str
    biosignal: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "core_metrics": self.core_metrics,
            "extensions": self.extensions,
            "basin_detection": self.basin_detection,
            "export_schema": self.export_schema,
            "affective": self.affective,
            "biosignal": self.biosignal
        }

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "AnalysisVersions":
        return cls(
            core_metrics=d.get("core_metrics", "unknown"),
            extensions=d.get("extensions", "unknown"),
            basin_detection=d.get("basin_detection", "unknown"),
            export_schema=d.get("export_schema", "unknown"),
            affective=d.get("affective", "unknown"),
            biosignal=d.get("biosignal", "unknown")
        )

    def __str__(self) -> str:
        return (f"core_metrics={self.core_metrics}, extensions={self.extensions}, "
                f"basin={self.basin_detection}, export={self.export_schema}")


def get_current_versions() -> AnalysisVersions:
    """Get current versions of all analysis modules."""
    return AnalysisVersions(
        core_metrics=CORE_METRICS_VERSION,
        extensions=EXTENSIONS_VERSION,
        basin_detection=BASIN_DETECTION_VERSION,
        export_schema=EXPORT_SCHEMA_VERSION,
        affective=AFFECTIVE_VERSION,
        biosignal=BIOSIGNAL_VERSION
    )


def get_versions_dict() -> Dict[str, str]:
    """Get current versions as a dictionary (for embedding in exports)."""
    return get_current_versions().to_dict()


# =============================================================================
# VERSION COMPARISON UTILITIES
# =============================================================================

def parse_version(version_str: str) -> tuple:
    """Parse version string to tuple for comparison."""
    try:
        parts = version_str.split(".")
        return tuple(int(p) for p in parts)
    except (ValueError, AttributeError):
        return (0, 0, 0)


def version_gte(v1: str, v2: str) -> bool:
    """Check if v1 >= v2."""
    return parse_version(v1) >= parse_version(v2)


def version_lt(v1: str, v2: str) -> bool:
    """Check if v1 < v2."""
    return parse_version(v1) < parse_version(v2)


def check_compatibility(session_versions: Dict[str, str]) -> Dict[str, dict]:
    """
    Check if session versions are compatible with current analysis.

    Returns dict with compatibility info for each module:
    {
        "core_metrics": {
            "session": "1.0.0",
            "current": "1.1.0",
            "compatible": True,
            "note": "Session uses pre-fix metrics"
        },
        ...
    }
    """
    current = get_versions_dict()
    result = {}

    for module, current_ver in current.items():
        session_ver = session_versions.get(module, "unknown")

        if session_ver == "unknown":
            compatible = False
            note = "No version info in session (pre-versioning export)"
        elif session_ver == current_ver:
            compatible = True
            note = "Exact match"
        elif version_lt(session_ver, current_ver):
            compatible = True  # Can read older formats
            note = f"Session uses older version ({session_ver})"
        else:
            compatible = False
            note = f"Session uses newer version ({session_ver}) than current ({current_ver})"

        result[module] = {
            "session": session_ver,
            "current": current_ver,
            "compatible": compatible,
            "note": note
        }

    return result


def needs_reanalysis(session_versions: Dict[str, str],
                     modules: Optional[list] = None) -> bool:
    """
    Check if a session should be re-analyzed with current versions.

    Args:
        session_versions: Version dict from session metadata
        modules: Specific modules to check (default: all)

    Returns:
        True if any specified module has changed since session was created
    """
    if modules is None:
        modules = ["core_metrics", "extensions", "basin_detection"]

    current = get_versions_dict()

    for module in modules:
        session_ver = session_versions.get(module, "unknown")
        current_ver = current.get(module, "unknown")

        if session_ver == "unknown" or session_ver != current_ver:
            return True

    return False


# =============================================================================
# MIGRATION HELPERS
# =============================================================================

def infer_versions_from_date(export_date: str) -> Dict[str, str]:
    """
    Infer likely versions from export date for pre-versioning sessions.

    This is a best-effort heuristic for sessions exported before
    version tracking was added.
    """
    from datetime import datetime

    try:
        dt = datetime.fromisoformat(export_date.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        # Can't parse date, assume oldest versions
        return {
            "core_metrics": "1.0.0",
            "extensions": "1.0.0",
            "basin_detection": "1.0.0",
            "export_schema": "1.0.0",
            "affective": "1.0.0",
            "biosignal": "1.0.0"
        }

    # Key dates for version changes
    # 2025-12-08: Core metrics fixes applied
    # 2025-12-11: Basin detection v2, export schema v2

    from datetime import timezone
    dt_naive = dt.replace(tzinfo=None) if dt.tzinfo else dt

    fix_date = datetime(2025, 12, 8)
    basin_v2_date = datetime(2025, 12, 11)

    if dt_naive < fix_date:
        core_ver = "1.0.0"
    else:
        core_ver = "1.1.0"

    if dt_naive < basin_v2_date:
        basin_ver = "1.0.0"
        ext_ver = "1.0.0"
        export_ver = "1.0.0"
    else:
        basin_ver = "2.0.0"
        ext_ver = "1.1.0"
        export_ver = "2.0.0"

    return {
        "core_metrics": core_ver,
        "extensions": ext_ver,
        "basin_detection": basin_ver,
        "export_schema": export_ver,
        "affective": "1.0.0",
        "biosignal": "1.0.0"
    }
