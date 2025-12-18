#!/usr/bin/env python3
"""
Validate refined basin detection against known sessions.

Expected results:
- Echo session (Sonnet 4): Should show Cognitive Mimicry or Reflexive Performance
- Collective solipsism (ChatGPT): Should show Collaborative Inquiry

Run from semantic-climate-phase-space directory:
    python scripts/validate_basin_refinement.py

Updated for Phase 3: Uses modular components (basins.py, substrates.py) instead
of the deprecated extensions.py methods.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Phase 3: Import from modular components
from src import BasinDetector, compute_dialogue_context


def extract_session_context(session_data: dict) -> dict:
    """Extract dialogue context from a session for basin detection."""
    conversation = session_data.get('conversation', [])

    turn_texts = []
    turn_speakers = []

    for turn in conversation:
        turn_texts.append(turn.get('text', ''))
        turn_speakers.append(turn.get('speaker', 'unknown'))

    return {
        'turn_texts': turn_texts,
        'turn_speakers': turn_speakers
    }


def analyze_session(filepath: Path, basin_detector: BasinDetector) -> dict:
    """Analyze a session and return basin detection results."""
    with open(filepath) as f:
        data = json.load(f)

    context = extract_session_context(data)

    # Compute dialogue context using module function
    dialogue_context = compute_dialogue_context(
        turn_texts=context['turn_texts'],
        turn_speakers=context['turn_speakers'],
        trajectory_metrics=None,  # Would need to compute from embeddings
        coherence_pattern='transitional',  # Default
        hedging_density=0.0  # Would need affective substrate
    )

    # For now, just show what we extracted
    return {
        'filepath': str(filepath.name),
        'turn_count': len(context['turn_texts']),
        'dialogue_context': dialogue_context
    }


def main():
    basin_detector = BasinDetector()

    # Test with Echo session
    echo_path = Path('sessions/sc-anthropic-claude-sonnet-4-20250514-2025-12-10T23-34-26-ece0e9a4.json')
    if echo_path.exists():
        print("=== Echo Session (Sonnet 4) ===")
        result = analyze_session(echo_path, basin_detector)
        print(f"Turn count: {result['turn_count']}")
        print(f"Turn length ratio (AI/human): {result['dialogue_context']['turn_length_ratio']:.2f}")
        print(f"Coherence pattern: {result['dialogue_context']['coherence_pattern']}")
        print()
    else:
        print(f"Echo session not found at {echo_path}")

    # Test with Collective solipsism from ChatGPT export
    cs_path = Path('../ChatGPT-Export/2025-12-08-Export/analysis/2023-07-29__CI-1_Collective_solipsism__23turns_analysis.json')
    if cs_path.exists():
        print("=== Collective Solipsism Analysis ===")
        with open(cs_path) as f:
            analysis = json.load(f)
        print(f"Final basin: {analysis['final_attractor']['name']}")
        print(f"Confidence: {analysis['final_attractor']['confidence']:.3f}")
        print(f"Trajectory windows: {len(analysis['trajectory'])}")

        # Show trajectory of basins
        print("\nBasin trajectory:")
        for window in analysis['trajectory']:
            basin = window['attractor_basin']
            coherence = window.get('coherence', {})
            print(f"  {window['turn_range']}: {basin['name']} ({basin['confidence']:.2f}) - {coherence.get('pattern', 'unknown')}")
    else:
        print(f"Collective solipsism analysis not found at {cs_path}")

    print("\n=== Basin Detection Test ===")

    # Test the three new basins with synthetic data
    test_cases = [
        {
            'name': 'Cognitive Mimicry (low hedging, AI dominates)',
            'psi_vector': {'psi_semantic': 0.5, 'psi_temporal': 0.5, 'psi_affective': 0.1, 'psi_biosignal': 0.0},
            'raw_metrics': {'delta_kappa': 0.5},  # High enough to not be sycophantic
            'dialogue_context': {
                'hedging_density': 0.005,  # Low
                'turn_length_ratio': 3.0,  # AI dominates
                'delta_kappa_variance': 0.003,  # Smooth
                'coherence_pattern': 'transitional'
            }
        },
        {
            'name': 'Collaborative Inquiry (hedging, balanced turns)',
            'psi_vector': {'psi_semantic': 0.5, 'psi_temporal': 0.5, 'psi_affective': 0.1, 'psi_biosignal': 0.0},
            'raw_metrics': {'delta_kappa': 0.5},
            'dialogue_context': {
                'hedging_density': 0.03,  # Present
                'turn_length_ratio': 1.2,  # Balanced
                'delta_kappa_variance': 0.02,  # Oscillating
                'coherence_pattern': 'breathing'
            }
        },
        {
            'name': 'Reflexive Performance (moderate hedging, scripted)',
            'psi_vector': {'psi_semantic': 0.5, 'psi_temporal': 0.5, 'psi_affective': 0.1, 'psi_biosignal': 0.0},
            'raw_metrics': {'delta_kappa': 0.5},
            'dialogue_context': {
                'hedging_density': 0.02,  # Moderate
                'turn_length_ratio': 2.0,  # AI dominates
                'delta_kappa_variance': 0.01,  # Scripted oscillation
                'coherence_pattern': 'transitional'
            }
        },
        {
            'name': 'Sycophantic Convergence (low delta_kappa)',
            'psi_vector': {'psi_semantic': 0.5, 'psi_temporal': 0.5, 'psi_affective': 0.1, 'psi_biosignal': 0.0},
            'raw_metrics': {'delta_kappa': 0.2},  # Low - flat trajectory
            'dialogue_context': {
                'hedging_density': 0.005,
                'turn_length_ratio': 2.0,
                'delta_kappa_variance': 0.002,
                'coherence_pattern': 'locked'
            }
        }
    ]

    for case in test_cases:
        # Use BasinDetector.detect() method
        basin, confidence, metadata = basin_detector.detect(
            psi_vector=case['psi_vector'],
            raw_metrics=case['raw_metrics'],
            dialogue_context=case['dialogue_context'],
            basin_history=None
        )
        expected = case['name'].split(' (')[0]
        status = "✓" if basin == expected else "✗"
        print(f"{status} {case['name']}")
        print(f"  Expected: {expected}")
        print(f"  Got: {basin} (confidence: {confidence:.3f})")
        print()


if __name__ == '__main__':
    main()
