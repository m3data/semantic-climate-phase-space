#!/usr/bin/env python3
"""
Batch analyze ChatGPT conversation exports for Semantic Climate patterns.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Usage:
    python batch_analyze_conversations.py /path/to/conversations-for-SC/
    python batch_analyze_conversations.py /path/to/single_conversation.json
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "semantic_climate_app" / "backend"))

from src import SemanticClimateAnalyzer
from semantic_climate_app.backend.embedding_service import EmbeddingService
from semantic_climate_app.backend.metrics_service import MetricsService


def load_conversation(filepath: Path) -> dict:
    """Load a preprocessed conversation JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_turns(conversation: dict) -> list[dict]:
    """
    Extract human/AI turns from conversation, filtering system messages.

    Returns list of dicts with 'speaker', 'text', 'timestamp'.
    """
    turns = []
    for turn in conversation.get('turns', []):
        role = turn.get('role')
        content = turn.get('content', '').strip()

        # Skip system messages and empty content
        if role == 'system' or not content:
            continue

        speaker = 'human' if role == 'user' else 'ai'
        turns.append({
            'speaker': speaker,
            'text': content,
            'timestamp': turn.get('timestamp')
        })

    return turns


def extract_turn_data(turns: list[dict]) -> tuple[list[str], list[str]]:
    """
    Extract parallel lists of texts and speakers from turn dicts.

    Args:
        turns: List of turn dicts with 'speaker', 'text' keys

    Returns:
        tuple: (texts, speakers) parallel lists
    """
    texts = [t['text'] for t in turns]
    speakers = [t['speaker'] for t in turns]
    return texts, speakers


def analyze_conversation(
    conversation: dict,
    embedding_service: EmbeddingService,
    metrics_service: MetricsService,
    min_turns: int = 10
) -> dict:
    """
    Run full SC analysis on a conversation.

    Returns dict with metadata, metrics trajectory, mode classifications.
    """
    turns = extract_turns(conversation)

    if len(turns) < min_turns:
        return {
            'status': 'skipped',
            'reason': f'insufficient_turns ({len(turns)} < {min_turns})',
            'turn_count': len(turns)
        }

    # Extract texts and speakers
    texts, speakers = extract_turn_data(turns)

    # Generate embeddings for all turns
    embeddings = embedding_service.embed_batch(texts)

    # Calculate metrics at different windows through the conversation
    # This gives us the trajectory over time
    metrics_trajectory = []
    window_size = min_turns
    step_size = max(1, (len(turns) - window_size) // 10)  # ~10 snapshots

    previous_metrics = None

    # Accumulate trajectory metrics for Δκ variance calculation
    accumulated_trajectory_metrics = []

    for i in range(0, len(turns) - window_size + 1, step_size):
        window_end = i + window_size
        window_embeddings = list(embeddings[i:window_end])
        window_texts = texts[i:window_end]
        window_speakers = speakers[i:window_end]

        # Calculate semantic shifts for coherence
        semantic_shifts = []
        for j in range(1, len(window_embeddings)):
            prev_emb = window_embeddings[j-1]
            curr_emb = window_embeddings[j]
            # Cosine distance
            sim = np.dot(prev_emb, curr_emb) / (np.linalg.norm(prev_emb) * np.linalg.norm(curr_emb))
            semantic_shifts.append(1.0 - sim)

        # Run analysis with dialogue context
        result = metrics_service.analyze(
            embeddings=window_embeddings,
            turn_texts=window_texts,
            turn_speakers=window_speakers,
            previous_metrics=previous_metrics,
            semantic_shifts=semantic_shifts,
            trajectory_metrics=accumulated_trajectory_metrics if accumulated_trajectory_metrics else None
        )

        # Store for trajectory calculation
        previous_metrics = {
            'delta_kappa': result['metrics']['delta_kappa'],
            'delta_h': result['metrics']['delta_h'],
            'alpha': result['metrics']['alpha']
        }

        # Accumulate trajectory metrics for refined basin detection
        accumulated_trajectory_metrics.append(previous_metrics.copy())

        metrics_trajectory.append({
            'window_start': i,
            'window_end': window_end,
            'turn_range': f"{i+1}-{window_end}",
            'metrics': result['metrics'],
            'mode': result['mode'],
            'coupling_mode': result['coupling_mode'],
            'psi_vector': result['psi_vector'],
            'psi_composite': result['psi_composite'],
            'attractor_basin': result['attractor_basin'],
            'coherence': result['coupling_mode'].get('coherence')
        })

    # Final full-conversation analysis with full dialogue context
    all_embeddings = list(embeddings)
    all_semantic_shifts = []
    for j in range(1, len(all_embeddings)):
        prev_emb = all_embeddings[j-1]
        curr_emb = all_embeddings[j]
        sim = np.dot(prev_emb, curr_emb) / (np.linalg.norm(prev_emb) * np.linalg.norm(curr_emb))
        all_semantic_shifts.append(1.0 - sim)

    final_result = metrics_service.analyze(
        embeddings=all_embeddings,
        turn_texts=texts,
        turn_speakers=speakers,
        previous_metrics=previous_metrics,
        semantic_shifts=all_semantic_shifts,
        trajectory_metrics=accumulated_trajectory_metrics
    )

    # Compute summary statistics
    modes_seen = [m['mode'] for m in metrics_trajectory]
    mode_counts = {}
    for m in modes_seen:
        mode_counts[m] = mode_counts.get(m, 0) + 1

    dominant_mode = max(mode_counts, key=mode_counts.get) if mode_counts else 'Unknown'

    # Trajectory direction (first vs last snapshot)
    if len(metrics_trajectory) >= 2:
        first = metrics_trajectory[0]['metrics']
        last = metrics_trajectory[-1]['metrics']
        dk_change = last['delta_kappa'] - first['delta_kappa']
        dh_change = last['delta_h'] - first['delta_h']
        alpha_change = last['alpha'] - first['alpha']
    else:
        dk_change = dh_change = alpha_change = 0.0

    return {
        'status': 'analyzed',
        'metadata': {
            'id': conversation.get('id'),
            'title': conversation.get('title'),
            'start_date': conversation.get('conversation_start_iso'),
            'turn_count': len(turns),
            'analyzed_at': datetime.now().isoformat()
        },
        'summary': {
            'dominant_mode': dominant_mode,
            'mode_distribution': mode_counts,
            'final_mode': final_result['mode'],
            'final_coupling': final_result['coupling_mode']['compound_label'],
            'epistemic_risk': final_result['coupling_mode']['epistemic_risk'],
            'risk_factors': final_result['coupling_mode']['risk_factors'],
            'trajectory_direction': {
                'delta_kappa_change': round(dk_change, 4),
                'delta_h_change': round(dh_change, 4),
                'alpha_change': round(alpha_change, 4),
                'overall': 'warming' if dk_change > 0.05 else ('cooling' if dk_change < -0.05 else 'stable')
            }
        },
        'final_metrics': final_result['metrics'],
        'final_psi': {
            'composite': final_result['psi_composite'],
            'vector': final_result['psi_vector']
        },
        'final_attractor': final_result['attractor_basin'],
        'final_coherence': final_result['coupling_mode'].get('coherence'),
        'trajectory': metrics_trajectory
    }


def main():
    parser = argparse.ArgumentParser(
        description='Batch analyze ChatGPT conversations for Semantic Climate patterns'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to conversation JSON file or directory of JSON files'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for analysis results (default: input_path/../analysis/)'
    )
    parser.add_argument(
        '--min-turns', '-m',
        type=int,
        default=10,
        help='Minimum turns required for analysis (default: 10)'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='all-mpnet-base-v2',
        help='Embedding model (default: all-mpnet-base-v2 via sentence-transformers)'
    )
    parser.add_argument(
        '--embedding-backend',
        type=str,
        default='sentence-transformers',
        choices=['sentence-transformers', 'ollama'],
        help='Embedding backend: sentence-transformers (default) or ollama'
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)

    # Determine files to process
    if input_path.is_file():
        files = [input_path]
        default_output = input_path.parent / 'analysis'
    elif input_path.is_dir():
        files = sorted(input_path.glob('*.json'))
        default_output = input_path.parent / 'analysis'
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(files)} conversation file(s)")
    print(f"Output directory: {output_dir}")
    print(f"Minimum turns: {args.min_turns}")
    print(f"Embedding: {args.embedding_model} via {args.embedding_backend}")
    print()

    # Initialize services
    print(f"Loading embedding model ({args.embedding_backend}: {args.embedding_model})...")
    embedding_service = EmbeddingService(
        model_name=args.embedding_model,
        backend=args.embedding_backend
    )
    metrics_service = MetricsService()
    print()

    # Process each conversation
    results = []

    for i, filepath in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing: {filepath.name}")

        try:
            conversation = load_conversation(filepath)
            result = analyze_conversation(
                conversation,
                embedding_service,
                metrics_service,
                min_turns=args.min_turns
            )

            result['source_file'] = filepath.name
            results.append(result)

            if result['status'] == 'analyzed':
                print(f"    Turns: {result['metadata']['turn_count']}")
                print(f"    Dominant mode: {result['summary']['dominant_mode']}")
                print(f"    Final: {result['summary']['final_coupling']}")
                print(f"    Trajectory: {result['summary']['trajectory_direction']['overall']}")
            else:
                print(f"    Skipped: {result['reason']}")

            # Save individual result
            output_file = output_dir / f"{filepath.stem}_analysis.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)

        except Exception as e:
            print(f"    Error: {e}")
            results.append({
                'status': 'error',
                'source_file': filepath.name,
                'error': str(e)
            })

    # Generate summary report
    print()
    print("=" * 60)
    print("BATCH ANALYSIS SUMMARY")
    print("=" * 60)

    analyzed = [r for r in results if r['status'] == 'analyzed']
    skipped = [r for r in results if r['status'] == 'skipped']
    errors = [r for r in results if r['status'] == 'error']

    print(f"Total files: {len(results)}")
    print(f"Analyzed: {len(analyzed)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Errors: {len(errors)}")
    print()

    if analyzed:
        # Mode distribution across all conversations
        all_modes = {}
        for r in analyzed:
            mode = r['summary']['dominant_mode']
            all_modes[mode] = all_modes.get(mode, 0) + 1

        print("Mode distribution (dominant mode per conversation):")
        for mode, count in sorted(all_modes.items(), key=lambda x: -x[1]):
            pct = count / len(analyzed) * 100
            print(f"  {mode}: {count} ({pct:.1f}%)")
        print()

        # Risk distribution
        risk_dist = {}
        for r in analyzed:
            risk = r['summary']['epistemic_risk']
            risk_dist[risk] = risk_dist.get(risk, 0) + 1

        print("Epistemic risk distribution:")
        for risk, count in sorted(risk_dist.items()):
            pct = count / len(analyzed) * 100
            print(f"  {risk}: {count} ({pct:.1f}%)")
        print()

        # Trajectory direction
        traj_dist = {}
        for r in analyzed:
            traj = r['summary']['trajectory_direction']['overall']
            traj_dist[traj] = traj_dist.get(traj, 0) + 1

        print("Trajectory direction distribution:")
        for traj, count in sorted(traj_dist.items()):
            pct = count / len(analyzed) * 100
            print(f"  {traj}: {count} ({pct:.1f}%)")

    # Save summary
    summary = {
        'generated_at': datetime.now().isoformat(),
        'input_path': str(input_path),
        'total_files': len(results),
        'analyzed': len(analyzed),
        'skipped': len(skipped),
        'errors': len(errors),
        'mode_distribution': all_modes if analyzed else {},
        'risk_distribution': risk_dist if analyzed else {},
        'trajectory_distribution': traj_dist if analyzed else {},
        'results': results
    }

    summary_file = output_dir / 'batch_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print(f"Summary saved to: {summary_file}")
    print(f"Individual analyses in: {output_dir}")


if __name__ == '__main__':
    main()
