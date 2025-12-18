#!/usr/bin/env python3
"""
Embedding Model Sensitivity Diagnostic

Compares local semantic motion sensitivity vs global conceptual structure
between embedding models (MiniLM vs nomic-embed-text).

Key questions:
1. Local sensitivity: How well does the model detect subtle turn-to-turn shifts?
2. Global structure: How well does it preserve topic/concept clusters?
3. Metric impact: How do Δκ, α, ΔH change between models?

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "semantic_climate_app" / "backend"))

from src import SemanticComplexityAnalyzer
from embedding_service import EmbeddingService


def load_session_texts(filepath: Path) -> List[str]:
    """Extract turn texts from a session export."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    texts = []
    for turn in data.get('conversation', []):
        text = turn.get('text', '').strip()
        if text:
            texts.append(text)
    return texts


def compute_semantic_velocities(embeddings: np.ndarray) -> np.ndarray:
    """Compute turn-to-turn cosine distances (semantic velocity)."""
    velocities = []
    for i in range(len(embeddings) - 1):
        e1, e2 = embeddings[i], embeddings[i + 1]
        cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        velocities.append(1.0 - cos_sim)  # distance = 1 - similarity
    return np.array(velocities)


def compute_local_sensitivity_metrics(velocities: np.ndarray) -> dict:
    """
    Compute metrics that characterize local motion sensitivity.

    A model sensitive to local motion will show:
    - Higher variance in velocities (detects subtle differences)
    - Lower autocorrelation (doesn't smooth over local changes)
    - Wider dynamic range
    """
    if len(velocities) < 3:
        return {'error': 'insufficient data'}

    # Variance: higher = more sensitive to local differences
    variance = float(np.var(velocities))

    # Dynamic range: max - min velocity
    dynamic_range = float(np.max(velocities) - np.min(velocities))

    # Coefficient of variation: std/mean (normalized variance)
    mean_vel = np.mean(velocities)
    cv = float(np.std(velocities) / mean_vel) if mean_vel > 0 else 0

    # Lag-1 autocorrelation: lower = more responsive to local changes
    # High autocorr means the model smooths over local shifts
    autocorr = float(np.corrcoef(velocities[:-1], velocities[1:])[0, 1])
    if np.isnan(autocorr):
        autocorr = 0.0

    # Entropy of velocity distribution (discretized)
    # Higher entropy = richer discrimination of shift magnitudes
    hist, _ = np.histogram(velocities, bins=10, density=True)
    hist = hist[hist > 0]  # Remove zeros
    entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))

    return {
        'mean_velocity': float(mean_vel),
        'velocity_variance': variance,
        'dynamic_range': dynamic_range,
        'coefficient_of_variation': cv,
        'lag1_autocorrelation': autocorr,
        'velocity_entropy': entropy
    }


def compute_global_structure_metrics(embeddings: np.ndarray, texts: List[str]) -> dict:
    """
    Compute metrics that characterize global conceptual structure.

    A model faithful to global structure will show:
    - Clear clustering of similar topics
    - High inter-cluster distance
    - Consistent semantic neighborhoods
    """
    if len(embeddings) < 6:
        return {'error': 'insufficient data'}

    # Compute pairwise cosine similarities
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    sim_matrix = normalized @ normalized.T

    # Average similarity (how tightly clustered overall)
    triu_indices = np.triu_indices(len(embeddings), k=1)
    avg_similarity = float(np.mean(sim_matrix[triu_indices]))

    # Similarity variance (spread of relationships)
    sim_variance = float(np.var(sim_matrix[triu_indices]))

    # First-last similarity (does it track conversation arc?)
    first_last_sim = float(sim_matrix[0, -1])

    # Average local neighborhood consistency
    # For each point, how similar is it to its immediate neighbors vs distant points?
    local_sims = []
    distant_sims = []
    for i in range(1, len(embeddings) - 1):
        local_sims.append((sim_matrix[i, i-1] + sim_matrix[i, i+1]) / 2)
        # Distant: average of points > 3 steps away
        distant_idx = [j for j in range(len(embeddings)) if abs(j - i) > 3]
        if distant_idx:
            distant_sims.append(np.mean([sim_matrix[i, j] for j in distant_idx]))

    local_vs_distant = float(np.mean(local_sims) - np.mean(distant_sims)) if distant_sims else 0

    return {
        'avg_pairwise_similarity': avg_similarity,
        'similarity_variance': sim_variance,
        'first_last_similarity': first_last_sim,
        'local_vs_distant_gap': local_vs_distant  # Higher = better local structure
    }


def compute_metric_impact(embeddings: np.ndarray) -> dict:
    """Compute the actual SC metrics (Δκ, α, ΔH) for comparison."""
    analyzer = SemanticComplexityAnalyzer()
    results = analyzer.calculate_all_metrics(list(embeddings))

    return {
        'delta_kappa': float(results['delta_kappa']),
        'alpha': float(results['alpha']),
        'delta_h': float(results['delta_h']),
        'complexity_detected': results['summary']['cognitive_complexity_detected']
    }


def run_diagnostic(session_path: Path):
    """Run full diagnostic comparison between embedding models."""

    print("=" * 70)
    print("EMBEDDING MODEL SENSITIVITY DIAGNOSTIC")
    print("=" * 70)
    print(f"\nSession: {session_path.name}")

    # Load texts
    texts = load_session_texts(session_path)
    print(f"Turns: {len(texts)}")
    print()

    # Initialize embedding services
    print("Loading embedding models...")
    print()

    models = {
        'MiniLM (384d)': EmbeddingService(
            model_name='all-MiniLM-L6-v2',
            backend='sentence-transformers'
        ),
        'MPNet (768d)': EmbeddingService(
            model_name='all-mpnet-base-v2',
            backend='sentence-transformers'
        ),
        'mxbai (1024d)': EmbeddingService(
            model_name='mxbai-embed-large',
            backend='ollama'
        ),
        'Nomic (768d)': EmbeddingService(
            model_name='nomic-embed-text:v1.5',
            backend='ollama'
        )
    }

    results = {}

    for model_name, service in models.items():
        print(f"\n{'─' * 70}")
        print(f"Model: {model_name}")
        print(f"{'─' * 70}")

        # Generate embeddings
        embeddings = service.embed_batch(texts)
        print(f"Embedding shape: {embeddings.shape}")

        # Compute velocities
        velocities = compute_semantic_velocities(embeddings)

        # Local sensitivity
        local = compute_local_sensitivity_metrics(velocities)
        print(f"\nLOCAL SENSITIVITY:")
        print(f"  Mean velocity:        {local['mean_velocity']:.4f}")
        print(f"  Velocity variance:    {local['velocity_variance']:.6f}")
        print(f"  Dynamic range:        {local['dynamic_range']:.4f}")
        print(f"  Coeff of variation:   {local['coefficient_of_variation']:.4f}")
        print(f"  Lag-1 autocorr:       {local['lag1_autocorrelation']:.4f}")
        print(f"  Velocity entropy:     {local['velocity_entropy']:.4f}")

        # Global structure
        global_s = compute_global_structure_metrics(embeddings, texts)
        print(f"\nGLOBAL STRUCTURE:")
        print(f"  Avg pairwise sim:     {global_s['avg_pairwise_similarity']:.4f}")
        print(f"  Similarity variance:  {global_s['similarity_variance']:.6f}")
        print(f"  First-last sim:       {global_s['first_last_similarity']:.4f}")
        print(f"  Local vs distant gap: {global_s['local_vs_distant_gap']:.4f}")

        # SC metrics impact
        sc = compute_metric_impact(embeddings)
        print(f"\nSC METRICS:")
        print(f"  Δκ (curvature):       {sc['delta_kappa']:.4f}")
        print(f"  α (DFA):              {sc['alpha']:.4f}")
        print(f"  ΔH (entropy shift):   {sc['delta_h']:.4f}")
        print(f"  Complexity detected:  {sc['complexity_detected']}")

        results[model_name] = {
            'local': local,
            'global': global_s,
            'sc_metrics': sc,
            'velocities': velocities.tolist()
        }

    # Comparison summary
    print(f"\n{'=' * 100}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 100}")

    # Dynamic model list
    model_keys = list(results.keys())
    short_names = {
        'MiniLM (384d)': 'MiniLM',
        'MPNet (768d)': 'MPNet',
        'mxbai (1024d)': 'mxbai',
        'Nomic (768d)': 'Nomic'
    }

    header = f"{'Metric':<26}"
    for mk in model_keys:
        header += f" {short_names.get(mk, mk):<10}"
    header += f" {'Best':<10}"
    print(f"\n{header}")
    print("-" * 100)

    comparisons = [
        ('Velocity variance', 'local', 'velocity_variance', 'high'),
        ('Dynamic range', 'local', 'dynamic_range', 'high'),
        ('Coeff of variation', 'local', 'coefficient_of_variation', 'high'),
        ('Lag-1 autocorr', 'local', 'lag1_autocorrelation', 'low_abs'),
        ('Velocity entropy', 'local', 'velocity_entropy', 'high'),
        ('Avg pairwise sim', 'global', 'avg_pairwise_similarity', 'mid'),
        ('Similarity variance', 'global', 'similarity_variance', 'high'),
        ('Local vs distant gap', 'global', 'local_vs_distant_gap', 'high'),
        ('Δκ (curvature)', 'sc_metrics', 'delta_kappa', 'info'),
        ('α (DFA)', 'sc_metrics', 'alpha', 'info'),
        ('ΔH (entropy)', 'sc_metrics', 'delta_h', 'info'),
    ]

    for label, category, key, prefer in comparisons:
        vals = {short_names.get(mk, mk): results[mk][category][key] for mk in model_keys}

        # Determine best
        if prefer == 'high':
            best = max(vals, key=vals.get)
        elif prefer == 'low_abs':
            abs_vals = {k: abs(v) for k, v in vals.items()}
            best = min(abs_vals, key=abs_vals.get)
        elif prefer == 'mid':
            target = 0.4
            dist_vals = {k: abs(v - target) for k, v in vals.items()}
            best = min(dist_vals, key=dist_vals.get)
        else:
            best = '—'

        row = f"{label:<26}"
        for mk in model_keys:
            row += f" {vals[short_names.get(mk, mk)]:<10.4f}"
        row += f" {best:<10}"
        print(row)

    # Interpretation
    print(f"\n{'=' * 100}")
    print("INTERPRETATION FOR SEMANTIC CLIMATE")
    print(f"{'=' * 100}")

    # Score each model on key criteria for SC
    model_names = [short_names.get(mk, mk) for mk in model_keys]
    scores = {name: 0 for name in model_names}

    def get_vals(category, key):
        return {short_names.get(mk, mk): results[mk][category][key] for mk in model_keys}

    def print_vals(vals):
        return ", ".join([f"{k}={v:.4f}" for k, v in vals.items()])

    # 1. Velocity variance (higher = more sensitive to local shifts)
    var_vals = get_vals('local', 'velocity_variance')
    best_var = max(var_vals, key=var_vals.get)
    scores[best_var] += 2  # Weight: important for Δκ, α
    print(f"\n1. Velocity variance (local sensitivity): {best_var} wins")
    print(f"   {print_vals(var_vals)}")

    # 2. Dynamic range (higher = better discrimination)
    dr_vals = get_vals('local', 'dynamic_range')
    best_dr = max(dr_vals, key=dr_vals.get)
    scores[best_dr] += 2
    print(f"\n2. Dynamic range: {best_dr} wins")
    print(f"   {print_vals(dr_vals)}")

    # 3. Local vs distant gap (higher = better structure preservation)
    gap_vals = get_vals('global', 'local_vs_distant_gap')
    best_gap = max(gap_vals, key=gap_vals.get)
    scores[best_gap] += 2
    print(f"\n3. Local vs distant gap (structure): {best_gap} wins")
    print(f"   {print_vals(gap_vals)}")

    # 4. α validity (0.5 = boundary condition = bad)
    alpha_vals = get_vals('sc_metrics', 'alpha')
    print(f"\n4. α (DFA) values:")
    print(f"   {print_vals(alpha_vals)}")
    for name, val in alpha_vals.items():
        if abs(val - 0.5) < 0.01:
            print(f"   ⚠ {name}: α=0.5 (boundary condition — insufficient variance for DFA)")
            scores[name] -= 2
        elif 0.6 <= val <= 1.0:
            print(f"   ✓ {name}: α in valid range ({val:.2f})")
            scores[name] += 1

    # 5. Avg similarity (mid-range preferred — not collapsed, not scattered)
    sim_vals = get_vals('global', 'avg_pairwise_similarity')
    print(f"\n5. Avg pairwise similarity (0.25-0.45 ideal):")
    print(f"   {print_vals(sim_vals)}")
    for name, val in sim_vals.items():
        if val > 0.5:
            print(f"   ⚠ {name}: High similarity — may collapse distinctions")
        elif val < 0.15:
            print(f"   ⚠ {name}: Low similarity — may fragment coherence")
        else:
            print(f"   ✓ {name}: Good similarity range")
            scores[name] += 1

    # Final verdict
    print(f"\n{'─' * 100}")
    print("SCORES (higher = better for Semantic Climate):")
    for name in model_names:
        bar = '█' * max(0, scores[name] + 3)  # Offset for visibility
        print(f"  {name:<10}: {scores[name]:>3}  {bar}")

    winner = max(scores, key=scores.get)
    print(f"\n→ RECOMMENDED: {winner}")

    recommendations = {
        'MPNet': "MPNet offers 768 dims with strong local sensitivity (sentence-transformers)",
        'MiniLM': "MiniLM preserves fine distinctions despite lower dimensionality",
        'mxbai': "mxbai-embed-large offers high dimensionality (1024) via Ollama",
        'Nomic': "Consider alternatives — Nomic may collapse local semantic motion"
    }
    print(f"  {recommendations.get(winner, '')}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare embedding model sensitivity')
    parser.add_argument(
        'session_path',
        type=str,
        nargs='?',
        default=None,
        help='Path to session JSON file'
    )

    args = parser.parse_args()

    if args.session_path:
        session_path = Path(args.session_path)
    else:
        # Default to a recent session
        default_path = Path(__file__).parent.parent / "research/session-exports/sc-llama3.2-latest-2025-12-07T17-05-15-65b0d9e7.json"
        if default_path.exists():
            session_path = default_path
        else:
            print("Usage: python embedding_sensitivity_diagnostic.py <session.json>")
            sys.exit(1)

    if not session_path.exists():
        print(f"Error: {session_path} not found")
        sys.exit(1)

    run_diagnostic(session_path)
