#!/usr/bin/env python3
"""
Re-analyze a session file with fixed Morgoulis metrics.

Usage:
    python tools/reanalyze_session.py <session_file> [--output <output_dir>]

Example:
    python tools/reanalyze_session.py sessions/sc-llama3.2-latest-2025-12-07.json
    python tools/reanalyze_session.py /path/to/session.json --output research/analysis/
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.core_metrics import SemanticComplexityAnalyzer
from src import SemanticClimateAnalyzer


def load_session(session_path: str) -> dict:
    """Load session file and detect structure."""
    with open(session_path, 'r') as f:
        session = json.load(f)

    # Detect conversation key
    if 'conversation' in session:
        conv_key = 'conversation'
    elif 'turns' in session:
        conv_key = 'turns'
    else:
        raise ValueError("Could not find conversation data in session file")

    return session, conv_key


def extract_texts(session: dict, conv_key: str) -> list:
    """Extract text content from conversation."""
    texts = []
    for turn in session[conv_key]:
        if 'text' in turn:
            texts.append(turn['text'])
        elif 'content' in turn:
            texts.append(turn['content'])
    return texts


def generate_embeddings(texts: list) -> np.ndarray:
    """Generate embeddings using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts)


def analyze_session(session_path: str, output_dir: str = None) -> dict:
    """
    Re-analyze a session with fixed metrics.

    Returns dict with old metrics, new metrics, and comparison.
    """
    session, conv_key = load_session(session_path)

    # Extract metadata
    metadata = session.get('metadata', {})
    model_name = metadata.get('model', 'unknown')
    session_id = metadata.get('session_id', Path(session_path).stem)

    # Get conversation
    conversation = session[conv_key]
    n_turns = len(conversation)

    print(f"Session: {session_id}")
    print(f"Model: {model_name}")
    print(f"Turns: {n_turns}")

    # Extract stored metrics if available
    stored_metrics = None
    if 'metrics_history' in session and session['metrics_history']:
        stored_metrics = session['metrics_history'][-1]

    # Generate embeddings
    print("\nGenerating embeddings...")
    texts = extract_texts(session, conv_key)
    embeddings = generate_embeddings(texts)
    print(f"Shape: {embeddings.shape}")

    # Run fixed analysis
    print("\nRunning fixed metrics analysis...")
    analyzer = SemanticComplexityAnalyzer()
    results = analyzer.calculate_all_metrics(embeddings)

    # Extended analysis
    climate_analyzer = SemanticClimateAnalyzer()
    turn_texts = []
    for turn in conversation:
        speaker = turn.get('speaker', turn.get('role', 'unknown'))
        text = turn.get('text', turn.get('content', ''))
        turn_texts.append(f"[{speaker}] {text}")

    coupling = climate_analyzer.compute_coupling_coefficient(embeddings, turn_texts=turn_texts)

    # Build results
    analysis = {
        'session_id': session_id,
        'model': model_name,
        'n_turns': n_turns,
        'analyzed_at': datetime.now().isoformat(),
        'fixed_metrics': {
            'delta_kappa': float(results['delta_kappa']),
            'delta_kappa_ci': [float(results['delta_kappa_ci'][0]), float(results['delta_kappa_ci'][1])],
            'alpha': float(results['alpha']),
            'alpha_ci': [float(results['alpha_ci'][0]), float(results['alpha_ci'][1])],
            'delta_h': float(results['delta_h']),
            'delta_h_ci': [float(results['delta_h_ci'][0]), float(results['delta_h_ci'][1])],
        },
        'extended': {
            'psi_composite': float(coupling['psi']),
            'attractor_basin': coupling['attractor_dynamics']['basin'],
            'attractor_confidence': float(coupling['attractor_dynamics']['confidence']),
        },
    }

    # Add comparison if stored metrics exist
    if stored_metrics:
        sm = stored_metrics['metrics']
        analysis['stored_metrics'] = {
            'delta_kappa': sm['delta_kappa'],
            'alpha': sm['alpha'],
            'delta_h': sm['delta_h'],
        }
        analysis['comparison'] = {
            'delta_kappa_diff': float(results['delta_kappa'] - sm['delta_kappa']),
            'alpha_diff': float(results['alpha'] - sm['alpha']),
            'delta_h_diff': float(results['delta_h'] - sm['delta_h']),
        }
        if 'attractor_basin' in stored_metrics:
            analysis['stored_metrics']['attractor_basin'] = stored_metrics['attractor_basin']['name']
            analysis['stored_metrics']['attractor_confidence'] = stored_metrics['attractor_basin']['confidence']

    # Print summary
    print("\n" + "=" * 60)
    print("FIXED METRICS")
    print("=" * 60)
    print(f"Δκ (Curvature): {results['delta_kappa']:.4f}")
    print(f"α  (Fractal):   {results['alpha']:.4f}")
    print(f"ΔH (Entropy):   {results['delta_h']:.4f}")
    print(f"\nΨ composite:    {coupling['psi']:.4f}")
    print(f"Attractor:      {coupling['attractor_dynamics']['basin']} (conf={coupling['attractor_dynamics']['confidence']:.2f})")

    if stored_metrics:
        print("\n" + "-" * 60)
        print("COMPARISON (OLD → FIXED)")
        print("-" * 60)
        print(f"Δκ: {sm['delta_kappa']:.4f} → {results['delta_kappa']:.4f} ({results['delta_kappa'] - sm['delta_kappa']:+.4f})")
        print(f"α:  {sm['alpha']:.4f} → {results['alpha']:.4f} ({results['alpha'] - sm['alpha']:+.4f})")
        print(f"ΔH: {sm['delta_h']:.4f} → {results['delta_h']:.4f} ({results['delta_h'] - sm['delta_h']:+.4f})")

    # Save if output dir specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        date_str = datetime.now().strftime('%Y-%m-%d')
        output_file = output_path / f"reanalysis_{session_id}_{date_str}.json"

        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description='Re-analyze a session with fixed Morgoulis metrics'
    )
    parser.add_argument('session_file', help='Path to session JSON file')
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')

    args = parser.parse_args()

    # Suppress tokenizers warning
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    try:
        analysis = analyze_session(args.session_file, args.output)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
