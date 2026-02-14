"""
Semantic Climate Phase Space - Core Library

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms. Non-commercial use permitted for research,
education, and community projects. Commercial use requires permission.

This package provides tools for measuring cognitive complexity in AI dialogue
systems, extending Morgoulis (2025) 4D Semantic Coupling Framework.

Core Metrics (Morgoulis, 2025, MIT License):
    - Semantic Curvature (Δκ): Trajectory non-linearity
    - Fractal Similarity Score (α): Self-organizing patterns via DFA
    - Entropy Shift (ΔH): Semantic reorganization

Extensions (This Project, ESL-A License):
    - Vector Ψ representation (semantic, temporal, affective, biosignal)
    - Attractor basin detection with hysteresis-aware history
    - Trajectory dynamics and integrity computation
    - Semantic Climate Model interpretation

Usage:
    from src import SemanticComplexityAnalyzer

    analyzer = SemanticComplexityAnalyzer()
    results = analyzer.calculate_all_metrics(embeddings)

    # Extended analyzer with trajectory integrity
    from src import SemanticClimateAnalyzer

    climate_analyzer = SemanticClimateAnalyzer(track_history=True, compute_integrity=True)
    result = climate_analyzer.compute_coupling_coefficient(embeddings, turn_texts=texts)

Attribution:
    Core metrics implementation by Daria Morgoulis (2025, MIT).
    https://github.com/daryamorgoulis/4d-semantic-coupling
"""

from .core_metrics import SemanticComplexityAnalyzer

# Phase 2: Import from modular components
from .trajectory import TrajectoryBuffer, TrajectoryStateVector, compute_trajectory_derivatives
from .substrates import (
    compute_semantic_substrate,
    compute_temporal_substrate,
    compute_affective_substrate,
    compute_biosignal_substrate,
    compute_dialogue_context,
)
from .basins import (
    BasinDetector,
    BasinHistory,
    SoftStateInference,
    HysteresisConfig,
    detect_attractor_basin,
    generate_movement_annotation,
)
from .integrity import IntegrityAnalyzer, TransformationDetector
from .analyzer import SemanticClimateAnalyzer

# Note: LegacySemanticClimateAnalyzer removed from __init__.py to allow deprecation
# warning to trigger. Tests that need it should import directly:
#   from src.extensions import SemanticClimateAnalyzer as LegacySemanticClimateAnalyzer

from .api import (
    semantic_curvature,
    semantic_curvature_ci,
    dfa_alpha,
    entropy_shift,
    icc_oneway_random,
    icc_bootstrap_ci,
    bland_altman,
    all_pairs_bland_altman,
    cosine_sim,
    bootstrap_ci,
)
from .schema import (
    get_current_versions,
    get_versions_dict,
    check_compatibility,
    needs_reanalysis,
    infer_versions_from_date,
    AnalysisVersions,
    CORE_METRICS_VERSION,
    EXTENSIONS_VERSION,
    BASIN_DETECTION_VERSION,
    EXPORT_SCHEMA_VERSION,
)

__all__ = [
    # Class-based API (production)
    'SemanticComplexityAnalyzer',
    'SemanticClimateAnalyzer',
    'TrajectoryBuffer',
    'TrajectoryStateVector',
    # New modular components (Phase 2)
    'BasinDetector',
    'BasinHistory',
    'IntegrityAnalyzer',
    'TransformationDetector',
    # Movement-preserving classification (v0.3.0)
    'SoftStateInference',
    'HysteresisConfig',
    'generate_movement_annotation',
    # Standalone functions
    'compute_trajectory_derivatives',
    'compute_semantic_substrate',
    'compute_temporal_substrate',
    'compute_affective_substrate',
    'compute_biosignal_substrate',
    'compute_dialogue_context',
    'detect_attractor_basin',
    # Legacy (deprecated) - removed from __init__.py, import directly from extensions.py
    # Function-based API (testing/research)
    'semantic_curvature',
    'semantic_curvature_ci',
    'dfa_alpha',
    'entropy_shift',
    # Statistical utilities
    'icc_oneway_random',
    'icc_bootstrap_ci',
    'bland_altman',
    'all_pairs_bland_altman',
    # Helpers
    'cosine_sim',
    'bootstrap_ci',
    # Schema versioning
    'get_current_versions',
    'get_versions_dict',
    'check_compatibility',
    'needs_reanalysis',
    'infer_versions_from_date',
    'AnalysisVersions',
    'CORE_METRICS_VERSION',
    'EXTENSIONS_VERSION',
    'BASIN_DETECTION_VERSION',
    'EXPORT_SCHEMA_VERSION',
]

__version__ = '0.4.0'
__author__ = 'Semantic Climate Phase Space Project'
__credits__ = ['Daria Morgoulis (core metrics)']
