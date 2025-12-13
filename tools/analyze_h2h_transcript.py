#!/usr/bin/env python3
"""
Human-to-human transcript analyzer for semantic coupling analysis.

Parses multi-participant conversation transcripts and calculates coupling metrics
using the fixed Morgoulis metrics (v1.1.0).

Usage:
    python tools/analyze_h2h_transcript.py <transcript.md> [--output <dir>]

Transcript format (markdown):
    **Speaker Name** *[timestamp]*: message text
    **Another Speaker** *[timestamp]*: response text
"""

import sys
import os
import json
import re
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core_metrics import SemanticComplexityAnalyzer
from src.extensions import SemanticClimateAnalyzer
from src.schema import get_current_versions, CORE_METRICS_VERSION
from sentence_transformers import SentenceTransformer


class H2HTranscriptAnalyzer:
    """Analyze human-to-human conversation transcripts for semantic coupling."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize with embedding model."""
        print(f"Loading embedding model: {embedding_model}...")
        self.embedder = SentenceTransformer(embedding_model)
        self.analyzer = SemanticClimateAnalyzer()
        self.embedding_model = embedding_model
        print(f"✓ Ready to analyze (core_metrics v{CORE_METRICS_VERSION})")

    def parse_transcript(self, transcript_text: str) -> list:
        """
        Parse human-to-human transcript into turns.

        Expected format (markdown):
        **Speaker Name** *[timestamp]*: message text
        **Another Speaker** *[timestamp]*: response text
        ...

        Returns:
            List of dicts with 'speaker', 'timestamp', and 'text'
        """
        turns = []
        lines = transcript_text.strip().split('\n')

        # Pattern: **Speaker Name** *[timestamp]*: message
        pattern = r'\*\*([^*]+)\*\*\s+\*\[([^\]]+)\]\*:\s*(.+)'

        current_turn = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            match = re.match(pattern, line)
            if match:
                # Save previous turn if exists
                if current_turn:
                    turns.append(current_turn)

                # Start new turn
                speaker = match.group(1).strip()
                timestamp = match.group(2).strip()
                text = match.group(3).strip()

                current_turn = {
                    'speaker': speaker,
                    'timestamp': timestamp,
                    'text': text
                }
            else:
                # Continuation of current turn
                if current_turn and line:
                    current_turn['text'] += ' ' + line

        # Save final turn
        if current_turn:
            turns.append(current_turn)

        return turns

    def generate_embeddings(self, turns: list) -> np.ndarray:
        """Generate embeddings for all turn texts."""
        texts = [turn['text'] for turn in turns]
        print(f"Generating embeddings for {len(texts)} turns...")
        embeddings = self.embedder.encode(texts)
        print(f"✓ Generated {embeddings.shape} embeddings")
        return embeddings

    def analyze(self, embeddings: np.ndarray, turns: list) -> dict:
        """Calculate semantic coupling metrics using fixed v1.1.0 metrics."""
        print(f"Analyzing {len(embeddings)} turns...")

        if len(embeddings) < 10:
            print("⚠️  Warning: Less than 10 turns, metrics may not be meaningful")

        # Use the SemanticClimateAnalyzer which uses fixed metrics
        turn_texts = [turn['text'] for turn in turns]

        # Calculate coupling coefficient (includes all metrics)
        psi_results = self.analyzer.compute_coupling_coefficient(
            dialogue_embeddings=embeddings,
            turn_texts=turn_texts
        )

        # Extract core metrics (from raw_metrics key in extensions output)
        results = {
            'delta_kappa': psi_results['raw_metrics']['delta_kappa'],
            'delta_h': psi_results['raw_metrics']['delta_h'],
            'alpha': psi_results['raw_metrics']['alpha'],
            'psi': psi_results
        }

        # Add coupling mode classification
        dk = results['delta_kappa']
        dh = results['delta_h']
        alpha = results['alpha']

        if dk < 0.20 and dh < 0.08:
            mode = 'Sycophantic'
            mode_icon = '🟡'
        elif 0.35 <= dk <= 0.65 and 0.08 <= dh <= 0.18 and alpha >= 0.45:
            mode = 'Resonant'
            mode_icon = '🟢'
        elif dh > 0.25 or (dk > 0.65 and alpha < 0.40):
            mode = 'Chaotic'
            mode_icon = '🔴'
        elif dk > 0.70 and dh < 0.05:
            mode = 'Contemplative'
            mode_icon = '🔵'
        else:
            mode = 'Exploratory'
            mode_icon = '🟠'

        results['coupling_mode'] = mode
        results['coupling_icon'] = mode_icon

        # Calculate participant statistics
        speakers = {}
        for turn in turns:
            speaker = turn['speaker']
            if speaker not in speakers:
                speakers[speaker] = 0
            speakers[speaker] += 1

        results['participant_stats'] = speakers

        return results

    def format_results(self, results: dict, turn_count: int, participants: dict) -> str:
        """Format results for display."""
        output = []
        output.append("\n" + "="*60)
        output.append("HUMAN-TO-HUMAN SEMANTIC COUPLING ANALYSIS")
        output.append(f"(core_metrics v{CORE_METRICS_VERSION})")
        output.append("="*60)
        output.append(f"\nTurns analyzed: {turn_count}")
        output.append(f"Participants: {len(participants)}")

        for speaker, count in sorted(participants.items(), key=lambda x: -x[1]):
            output.append(f"  - {speaker}: {count} turns")

        output.append(f"\nCoupling Mode: {results['coupling_icon']} {results['coupling_mode']}")

        output.append("\n" + "-"*60)
        output.append("SEMANTIC CLIMATE:")
        output.append("-"*60)
        output.append(f"  Δκ (Curvature):    {results['delta_kappa']:.4f}")
        output.append(f"  ΔH (Entropy):      {results['delta_h']:.4f}")
        output.append(f"  α  (Fractal):      {results['alpha']:.4f}")

        # Display Ψ (Coupling Coefficient) - Vector format
        if 'psi' in results:
            psi = results['psi']
            output.append("\n" + "-"*60)
            output.append("COUPLING COEFFICIENT (Ψ):")
            output.append("-"*60)
            output.append(f"  Ψ (composite):     {psi['psi']:+.3f}")

            # Display 4D vector components
            pos = psi['psi_state']['position']
            output.append("\n  Phase-Space Vector:")
            output.append(f"    Ψ_semantic:      {pos['psi_semantic']:+.3f}")
            output.append(f"    Ψ_temporal:      {pos['psi_temporal']:+.3f}")
            if pos['psi_affective'] is not None:
                output.append(f"    Ψ_affective:     {pos['psi_affective']:+.3f}")
            else:
                output.append(f"    Ψ_affective:     None")
            output.append(f"    Ψ_biosignal:     {pos['psi_biosignal']}")

            # Display attractor basin
            att = psi['attractor_dynamics']
            output.append(f"\n  Attractor Basin:   {att['basin']}")
            output.append(f"  Basin Confidence:  {att['confidence']:.1%}")

        output.append("\n" + "-"*60)
        output.append("INTERPRETATION:")
        output.append("-"*60)

        # Interpret metrics
        output.append(f"\n  Semantic curvature (Δκ = {results['delta_kappa']:.3f}):")
        if results['delta_kappa'] < 0.20:
            output.append("    Low - Linear topic progression")
        elif results['delta_kappa'] <= 0.50:
            output.append("    Moderate - Natural conversational flow")
        elif results['delta_kappa'] <= 0.70:
            output.append("    Elevated - Rich exploratory dialogue")
        else:
            output.append("    High - Dense philosophical discourse")

        output.append(f"\n  Entropy shift (ΔH = {results['delta_h']:.3f}):")
        if results['delta_h'] < 0.08:
            output.append("    Low - Topically coherent discussion")
        elif results['delta_h'] <= 0.18:
            output.append("    Moderate - Healthy conceptual diversity")
        else:
            output.append("    High - Multiple diverging threads")

        output.append(f"\n  Fractal similarity (α = {results['alpha']:.3f}):")
        if results['alpha'] < 0.45:
            output.append("    Low - Fragmented patterns")
        elif results['alpha'] <= 0.70:
            output.append("    Moderate - Organic conversational rhythm")
        else:
            output.append("    High - Strong self-similarity across scales")

        output.append("\n" + "="*60)
        output.append("Note: These metrics use fixed Morgoulis implementation (v1.1.0)")
        output.append("Human-to-human provides baseline for human-AI comparison.")
        output.append("="*60)

        return '\n'.join(output)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze human-to-human transcript for semantic coupling'
    )
    parser.add_argument('transcript', help='Path to transcript file (.md)')
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--json', action='store_true', help='Output JSON only')
    args = parser.parse_args()

    transcript_file = args.transcript

    if not os.path.exists(transcript_file):
        print(f"Error: File not found: {transcript_file}")
        sys.exit(1)

    # Read transcript
    with open(transcript_file, 'r') as f:
        transcript_text = f.read()

    # Analyze
    analyzer = H2HTranscriptAnalyzer()
    turns = analyzer.parse_transcript(transcript_text)

    print(f"\nParsed {len(turns)} turns from transcript")

    if len(turns) < 3:
        print("Error: Need at least 3 turns for analysis")
        sys.exit(1)

    # Show participant distribution
    speakers = {}
    for turn in turns:
        speaker = turn['speaker']
        if speaker not in speakers:
            speakers[speaker] = 0
        speakers[speaker] += 1

    print(f"Found {len(speakers)} participants:")
    for speaker, count in sorted(speakers.items(), key=lambda x: -x[1]):
        print(f"  - {speaker}: {count} turns")

    embeddings = analyzer.generate_embeddings(turns)
    results = analyzer.analyze(embeddings, turns)

    # Display results
    if not args.json:
        print(analyzer.format_results(results, len(turns), speakers))

    # Build JSON output with version tracking
    versions = get_current_versions()
    json_results = {
        'metadata': {
            'transcript_file': transcript_file,
            'analyzed_at': datetime.now().isoformat(),
            'turn_count': len(turns),
            'participant_count': len(speakers),
            'participants': speakers,
            'embedding_model': analyzer.embedding_model,
            'versions': versions.to_dict()
        },
        'metrics': {
            'delta_kappa': float(results['delta_kappa']),
            'delta_h': float(results['delta_h']),
            'alpha': float(results['alpha']),
            'coupling_mode': results['coupling_mode']
        }
    }

    # Add Ψ (Coupling Coefficient)
    if 'psi' in results:
        psi = results['psi']
        pos = psi['psi_state']['position']

        json_results['coupling_coefficient'] = {
            'psi': float(psi['psi']),
            'psi_vector': {
                'semantic': float(pos['psi_semantic']),
                'temporal': float(pos['psi_temporal']),
                'affective': float(pos['psi_affective']) if pos['psi_affective'] is not None else None,
                'biosignal': pos['psi_biosignal']
            },
            'attractor_dynamics': {
                'basin': psi['attractor_dynamics']['basin'],
                'confidence': float(psi['attractor_dynamics']['confidence']),
                'basin_stability': psi['attractor_dynamics']['basin_stability']
            }
        }

    # Determine output path
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / (Path(transcript_file).stem + '_h2h_analysis.json')
    else:
        output_file = Path(transcript_file).with_suffix('') / '_h2h_analysis.json'
        output_file = Path(str(transcript_file).rsplit('.', 1)[0] + '_h2h_analysis.json')

    # Save results
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    if args.json:
        print(json.dumps(json_results, indent=2))


if __name__ == '__main__':
    main()
