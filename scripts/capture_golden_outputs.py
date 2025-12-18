"""
Capture Golden Outputs for Regression Testing

This script captures the current behavior of SemanticClimateAnalyzer
before the modular refactor. The outputs are saved to tests/fixtures/
for comparison during Phase 2 testing.

Usage:
    python scripts/capture_golden_outputs.py

Output:
    tests/fixtures/golden_outputs.json
"""

import json
import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src import SemanticClimateAnalyzer, TrajectoryBuffer


def create_test_scenarios():
    """Create diverse test scenarios to capture current behavior."""

    np.random.seed(42)  # Reproducibility
    scenarios = []

    # === Scenario 1: Basic embeddings only ===
    scenarios.append({
        'name': 'basic_embeddings_only',
        'description': 'Minimal input - just embeddings',
        'inputs': {
            'dialogue_embeddings': np.random.randn(20, 384).tolist(),
        }
    })

    # === Scenario 2: Embeddings + turn texts (for affective substrate) ===
    hedging_texts = [
        "I think this is interesting.",
        "Maybe we could explore this further?",
        "I'm not sure about that approach.",
        "Perhaps there's another way.",
        "I feel uncertain about this.",
        "It seems like it could work.",
        "I guess we should try it.",
        "I wonder if that's right.",
        "Possibly we should reconsider.",
        "I'm not certain, but maybe.",
    ] * 2

    scenarios.append({
        'name': 'with_hedging_texts',
        'description': 'Embeddings + hedging-rich turn texts',
        'inputs': {
            'dialogue_embeddings': np.random.randn(20, 384).tolist(),
            'turn_texts': hedging_texts,
        }
    })

    # === Scenario 3: Confident texts (low hedging) ===
    confident_texts = [
        "This is definitely the right approach.",
        "The answer is clearly obvious.",
        "We absolutely must do this.",
        "There is no question about it.",
        "This will certainly work.",
        "I'm completely sure about this.",
        "Without doubt, this is correct.",
        "The solution is straightforward.",
        "Obviously, we should proceed.",
        "This is undoubtedly the way.",
    ] * 2

    scenarios.append({
        'name': 'with_confident_texts',
        'description': 'Embeddings + confident turn texts (low hedging)',
        'inputs': {
            'dialogue_embeddings': np.random.randn(20, 384).tolist(),
            'turn_texts': confident_texts,
        }
    })

    # === Scenario 4: With speakers for turn ratio ===
    scenarios.append({
        'name': 'with_speakers',
        'description': 'Embeddings + texts + speaker labels',
        'inputs': {
            'dialogue_embeddings': np.random.randn(20, 384).tolist(),
            'turn_texts': hedging_texts,
            'turn_speakers': ['human', 'ai'] * 10,
        }
    })

    # === Scenario 5: With biosignal data ===
    scenarios.append({
        'name': 'with_biosignal',
        'description': 'Embeddings + biosignal data',
        'inputs': {
            'dialogue_embeddings': np.random.randn(20, 384).tolist(),
            'biosignal_data': {'heart_rate': 75},
        }
    })

    # === Scenario 6: With trajectory history ===
    trajectory_history = [
        {'psi_semantic': 0.1, 'psi_temporal': 0.5, 'psi_affective': 0.0, 'psi_biosignal': None},
        {'psi_semantic': 0.2, 'psi_temporal': 0.5, 'psi_affective': 0.1, 'psi_biosignal': None},
        {'psi_semantic': 0.3, 'psi_temporal': 0.5, 'psi_affective': 0.15, 'psi_biosignal': None},
        {'psi_semantic': 0.35, 'psi_temporal': 0.5, 'psi_affective': 0.2, 'psi_biosignal': None},
        {'psi_semantic': 0.4, 'psi_temporal': 0.5, 'psi_affective': 0.25, 'psi_biosignal': None},
    ]

    scenarios.append({
        'name': 'with_trajectory_history',
        'description': 'Embeddings + trajectory history for dynamics',
        'inputs': {
            'dialogue_embeddings': np.random.randn(20, 384).tolist(),
            'trajectory_history': trajectory_history,
        }
    })

    # === Scenario 7: Full context ===
    scenarios.append({
        'name': 'full_context',
        'description': 'All inputs provided',
        'inputs': {
            'dialogue_embeddings': np.random.randn(20, 384).tolist(),
            'turn_texts': hedging_texts,
            'turn_speakers': ['human', 'ai'] * 10,
            'biosignal_data': {'heart_rate': 85},
            'trajectory_history': trajectory_history,
        }
    })

    # === Scenario 8: Short sequence (edge case) ===
    scenarios.append({
        'name': 'short_sequence',
        'description': 'Minimal viable sequence (6 turns)',
        'inputs': {
            'dialogue_embeddings': np.random.randn(6, 384).tolist(),
        }
    })

    # === Scenario 9: Medium sequence (skip very long for speed) ===
    scenarios.append({
        'name': 'medium_sequence',
        'description': 'Medium dialogue (30 turns)',
        'inputs': {
            'dialogue_embeddings': np.random.randn(30, 384).tolist(),
        }
    })

    # === Scenario 10: Pre-computed metrics ===
    scenarios.append({
        'name': 'with_precomputed_metrics',
        'description': 'Metrics provided, no embeddings needed for metrics',
        'inputs': {
            'dialogue_embeddings': np.random.randn(20, 384).tolist(),
            'metrics': {
                'delta_kappa': 0.42,
                'delta_h': 0.18,
                'alpha': 0.85,
            }
        }
    })

    return scenarios


def capture_trajectory_buffer_outputs():
    """Capture TrajectoryBuffer behavior for regression testing."""

    results = {}

    # Linear trajectory
    buffer = TrajectoryBuffer(timestep=1.0)
    for i in range(5):
        buffer.append({
            'psi_semantic': 0.1 * i,
            'psi_temporal': 0.5,
            'psi_affective': 0.0,
            'psi_biosignal': None
        })

    results['linear_trajectory'] = {
        'velocity': buffer.compute_velocity(),
        'acceleration': buffer.compute_acceleration(),
        'curvature': buffer.compute_curvature(),
        'segment_length': len(buffer.get_trajectory_segment() or []),
    }

    # Sinusoidal trajectory
    buffer2 = TrajectoryBuffer(timestep=1.0)
    for i in range(10):
        buffer2.append({
            'psi_semantic': float(np.sin(i * 0.5)),
            'psi_temporal': float(np.cos(i * 0.5)),
            'psi_affective': 0.1 * i,
            'psi_biosignal': None
        })

    results['sinusoidal_trajectory'] = {
        'velocity': buffer2.compute_velocity(),
        'acceleration': buffer2.compute_acceleration(),
        'curvature': buffer2.compute_curvature(),
    }

    return results


def main():
    """Capture golden outputs and save to fixtures."""

    print("Capturing golden outputs for regression testing...", flush=True)
    print("=" * 60, flush=True)

    # Create fixtures directory if needed
    fixtures_dir = project_root / 'tests' / 'fixtures'
    fixtures_dir.mkdir(exist_ok=True)

    # Initialize analyzer
    analyzer = SemanticClimateAnalyzer(random_state=42)

    # Capture scenario outputs
    scenarios = create_test_scenarios()
    golden_outputs = {
        'version': '0.1.x-pre-refactor',
        'description': 'Golden outputs captured before modular refactor',
        'scenarios': {},
        'trajectory_buffer': {},
    }

    for scenario in scenarios:
        print(f"\nCapturing: {scenario['name']}", flush=True)
        print(f"  {scenario['description']}", flush=True)

        # Convert lists back to numpy arrays
        inputs = scenario['inputs'].copy()
        if 'dialogue_embeddings' in inputs:
            inputs['dialogue_embeddings'] = np.array(inputs['dialogue_embeddings'])

        try:
            result = analyzer.compute_coupling_coefficient(**inputs)

            # Convert numpy types to JSON-serializable
            golden_outputs['scenarios'][scenario['name']] = {
                'inputs_hash': hash(str(scenario['inputs'])),  # For verification
                'psi': float(result['psi']),
                'psi_state': {
                    'position': {
                        k: float(v) if v is not None else None
                        for k, v in result['psi_state']['position'].items()
                    },
                    'velocity': result['psi_state']['velocity'],
                    'acceleration': result['psi_state']['acceleration'],
                },
                'attractor_dynamics': {
                    'basin': result['attractor_dynamics']['basin'],
                    'confidence': float(result['attractor_dynamics']['confidence']),
                },
                'raw_metrics': {
                    k: float(v) for k, v in result['raw_metrics'].items()
                },
                'dialogue_context': result.get('dialogue_context'),
                'substrate_details': {
                    'semantic': {
                        k: float(v) if isinstance(v, (int, float, np.floating)) else v
                        for k, v in result['substrate_details']['semantic'].items()
                        if k != 'raw_metrics'  # Avoid duplication
                    },
                    'temporal': {
                        k: float(v) if isinstance(v, (int, float, np.floating)) else v
                        for k, v in result['substrate_details']['temporal'].items()
                    },
                    'affective': {
                        'psi_affective': float(result['substrate_details']['affective']['psi_affective']),
                        'hedging_density': float(result['substrate_details']['affective']['hedging_density']),
                        'vulnerability_score': float(result['substrate_details']['affective']['vulnerability_score']),
                        'confidence_variance': float(result['substrate_details']['affective']['confidence_variance']),
                    },
                },
            }
            print(f"  -> Captured: psi={result['psi']:.4f}, basin={result['attractor_dynamics']['basin']}", flush=True)

        except Exception as e:
            print(f"  -> ERROR: {e}", flush=True)
            golden_outputs['scenarios'][scenario['name']] = {'error': str(e)}

    # Capture trajectory buffer outputs
    print("\nCapturing TrajectoryBuffer behavior...", flush=True)
    golden_outputs['trajectory_buffer'] = capture_trajectory_buffer_outputs()

    # Save to file
    output_path = fixtures_dir / 'golden_outputs.json'
    with open(output_path, 'w') as f:
        json.dump(golden_outputs, f, indent=2, default=str)

    print(f"\n{'=' * 60}", flush=True)
    print(f"Golden outputs saved to: {output_path}", flush=True)
    print(f"Captured {len(golden_outputs['scenarios'])} scenarios", flush=True)

    return golden_outputs


if __name__ == '__main__':
    main()
