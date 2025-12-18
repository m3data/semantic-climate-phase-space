"""
Coupling Signature Analysis: Cross-Correlation

Analyzes coupling between semantic (Δκ) and physiological (HR) systems
using cross-correlation with lag analysis.

Based on: research/theory/coupling-signatures.md (Section 2.1)

Usage:
    python tools/coupling_analysis.py <session_export.json>
    python tools/coupling_analysis.py <session_export.json> --plot
"""

import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import signal
from scipy import stats

# Import version tracking
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.schema import get_versions_dict, infer_versions_from_date


@dataclass
class TimeSeriesData:
    """Aligned time series for coupling analysis."""
    timestamps: np.ndarray  # Common timestamps (seconds from start)
    hr: np.ndarray          # Heart rate values
    delta_kappa: np.ndarray # Semantic curvature values
    sample_rate: float      # Effective sample rate (Hz)

    def __len__(self):
        return len(self.timestamps)


@dataclass
class CrossCorrelationResult:
    """Results from cross-correlation analysis."""
    lags: np.ndarray           # Lag values in samples
    lags_seconds: np.ndarray   # Lag values in seconds
    correlation: np.ndarray    # Cross-correlation values
    peak_lag: int              # Lag at maximum correlation (samples)
    peak_lag_seconds: float    # Lag at maximum correlation (seconds)
    peak_correlation: float    # Maximum correlation value
    null_threshold_95: float   # 95th percentile from null distribution
    null_threshold_99: float   # 99th percentile from null distribution
    is_significant_95: bool    # Peak exceeds 95% threshold
    is_significant_99: bool    # Peak exceeds 99% threshold
    interpretation: str        # Human-readable interpretation


def load_session(filepath: str) -> Tuple[Dict, Dict]:
    """
    Load session export and return metadata and data.

    Returns:
        (metadata, session_data)
    """
    with open(filepath) as f:
        data = json.load(f)

    metadata = data.get('metadata', {})

    # Check for version info, infer if missing
    if 'versions' not in metadata:
        export_date = metadata.get('exported_at', '')
        metadata['versions'] = infer_versions_from_date(export_date)
        metadata['versions_inferred'] = True

    return metadata, data


def extract_time_series(session_data: Dict) -> Optional[TimeSeriesData]:
    """
    Extract and align HR and Δκ time series from session data.

    The challenge: biosignal streams at ~1Hz, metrics computed every N turns.
    We need to align them on a common time axis.
    """
    biosignal = session_data.get('biosignal_stream', [])
    metrics_history = session_data.get('metrics_history', [])

    if not biosignal or not metrics_history:
        return None

    # Parse biosignal timestamps and HR
    bio_times = []
    bio_hr = []
    for sample in biosignal:
        ts = datetime.fromisoformat(sample['ts'].replace('Z', '+00:00'))
        bio_times.append(ts)
        bio_hr.append(sample['hr'])

    if not bio_times:
        return None

    # Convert to seconds from start
    start_time = bio_times[0]
    bio_seconds = np.array([(t - start_time).total_seconds() for t in bio_times])
    bio_hr = np.array(bio_hr)

    # Parse metrics timestamps and Δκ
    metric_times = []
    metric_dk = []
    for entry in metrics_history:
        ts = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
        metric_times.append(ts)
        dk = entry.get('metrics', {}).get('delta_kappa', 0)
        metric_dk.append(dk)

    if not metric_times:
        return None

    metric_seconds = np.array([(t - start_time).total_seconds() for t in metric_times])
    metric_dk = np.array(metric_dk)

    # Interpolate Δκ to biosignal timestamps
    # (metrics are sparse, biosignal is dense)
    if len(metric_seconds) < 2:
        return None

    # Use linear interpolation for Δκ
    dk_interpolated = np.interp(bio_seconds, metric_seconds, metric_dk)

    # Estimate sample rate from biosignal
    if len(bio_seconds) > 1:
        intervals = np.diff(bio_seconds)
        sample_rate = 1.0 / np.median(intervals)
    else:
        sample_rate = 1.0

    return TimeSeriesData(
        timestamps=bio_seconds,
        hr=bio_hr,
        delta_kappa=dk_interpolated,
        sample_rate=sample_rate
    )


def compute_cross_correlation(
    ts_data: TimeSeriesData,
    max_lag_seconds: float = 60.0,
    n_permutations: int = 1000
) -> CrossCorrelationResult:
    """
    Compute cross-correlation between HR and Δκ with significance testing.

    Args:
        ts_data: Aligned time series data
        max_lag_seconds: Maximum lag to consider (in seconds)
        n_permutations: Number of permutations for null distribution

    Returns:
        CrossCorrelationResult with correlation, lags, and significance
    """
    hr = ts_data.hr
    dk = ts_data.delta_kappa

    # Normalize (z-score) both signals
    hr_norm = (hr - np.mean(hr)) / (np.std(hr) + 1e-10)
    dk_norm = (dk - np.mean(dk)) / (np.std(dk) + 1e-10)

    # Compute cross-correlation
    # mode='full' gives correlation for all possible lags
    correlation = signal.correlate(hr_norm, dk_norm, mode='full')
    correlation = correlation / len(hr)  # Normalize

    # Compute lag axis
    n = len(hr)
    lags = np.arange(-(n-1), n)
    lags_seconds = lags / ts_data.sample_rate

    # Limit to max_lag
    max_lag_samples = int(max_lag_seconds * ts_data.sample_rate)
    center = n - 1
    lag_mask = np.abs(lags) <= max_lag_samples

    lags = lags[lag_mask]
    lags_seconds = lags_seconds[lag_mask]
    correlation = correlation[lag_mask]

    # Find peak
    peak_idx = np.argmax(np.abs(correlation))
    peak_lag = lags[peak_idx]
    peak_lag_seconds = lags_seconds[peak_idx]
    peak_correlation = correlation[peak_idx]

    # Null distribution via permutation
    null_peaks = []
    for _ in range(n_permutations):
        # Shuffle one signal
        hr_shuffled = np.random.permutation(hr_norm)
        null_corr = signal.correlate(hr_shuffled, dk_norm, mode='full')
        null_corr = null_corr / len(hr)
        null_corr = null_corr[lag_mask]
        null_peaks.append(np.max(np.abs(null_corr)))

    null_peaks = np.array(null_peaks)
    null_threshold_95 = np.percentile(null_peaks, 95)
    null_threshold_99 = np.percentile(null_peaks, 99)

    is_significant_95 = np.abs(peak_correlation) > null_threshold_95
    is_significant_99 = np.abs(peak_correlation) > null_threshold_99

    # Interpretation
    interpretation = interpret_correlation(
        peak_lag_seconds, peak_correlation,
        is_significant_95, is_significant_99
    )

    return CrossCorrelationResult(
        lags=lags,
        lags_seconds=lags_seconds,
        correlation=correlation,
        peak_lag=peak_lag,
        peak_lag_seconds=peak_lag_seconds,
        peak_correlation=peak_correlation,
        null_threshold_95=null_threshold_95,
        null_threshold_99=null_threshold_99,
        is_significant_95=is_significant_95,
        is_significant_99=is_significant_99,
        interpretation=interpretation
    )


def interpret_correlation(
    peak_lag_seconds: float,
    peak_correlation: float,
    sig_95: bool,
    sig_99: bool
) -> str:
    """Generate human-readable interpretation of cross-correlation results."""

    if not sig_95:
        return "No significant coupling detected (correlation within null distribution)"

    sig_level = "p < 0.01" if sig_99 else "p < 0.05"
    direction = "positive" if peak_correlation > 0 else "negative"

    if abs(peak_lag_seconds) < 1.0:
        # Near-zero lag
        timing = "synchronous (near-zero lag)"
        causal = "coincident variation, no clear causal direction"
    elif peak_lag_seconds > 0:
        # Positive lag: HR leads Δκ
        timing = f"HR leads Δκ by {abs(peak_lag_seconds):.1f}s"
        causal = "physiological state may influence semantic trajectory"
    else:
        # Negative lag: Δκ leads HR
        timing = f"Δκ leads HR by {abs(peak_lag_seconds):.1f}s"
        causal = "semantic content may influence physiological state"

    strength = "strong" if abs(peak_correlation) > 0.5 else \
               "moderate" if abs(peak_correlation) > 0.3 else "weak"

    return (
        f"Significant {strength} {direction} coupling detected ({sig_level})\n"
        f"Timing: {timing}\n"
        f"Interpretation: {causal}"
    )


def plot_correlogram(
    result: CrossCorrelationResult,
    output_path: Optional[str] = None,
    title: str = "Cross-Correlation: HR × Δκ"
):
    """Plot the cross-correlogram with significance thresholds."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot correlation
    ax.plot(result.lags_seconds, result.correlation, 'b-', linewidth=1.5, label='Cross-correlation')

    # Plot significance thresholds
    ax.axhline(result.null_threshold_95, color='orange', linestyle='--',
               label=f'95% threshold ({result.null_threshold_95:.3f})')
    ax.axhline(-result.null_threshold_95, color='orange', linestyle='--')
    ax.axhline(result.null_threshold_99, color='red', linestyle='--',
               label=f'99% threshold ({result.null_threshold_99:.3f})')
    ax.axhline(-result.null_threshold_99, color='red', linestyle='--')

    # Mark peak
    ax.axvline(result.peak_lag_seconds, color='green', linestyle=':', alpha=0.7)
    ax.plot(result.peak_lag_seconds, result.peak_correlation, 'go', markersize=10,
            label=f'Peak: r={result.peak_correlation:.3f} at {result.peak_lag_seconds:.1f}s')

    # Zero lag reference
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel('Lag (seconds)\n← Δκ leads | HR leads →')
    ax.set_ylabel('Cross-correlation')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add interpretation text
    sig_marker = "**" if result.is_significant_99 else "*" if result.is_significant_95 else ""
    ax.text(0.02, 0.98, f"Peak lag: {result.peak_lag_seconds:.1f}s {sig_marker}",
            transform=ax.transAxes, verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def analyze_session(
    filepath: str,
    max_lag_seconds: float = 60.0,
    n_permutations: int = 1000,
    plot: bool = False,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Run full cross-correlation analysis on a session export.

    Args:
        filepath: Path to session export JSON
        max_lag_seconds: Maximum lag to analyze
        n_permutations: Number of permutations for null model
        plot: Whether to generate plots
        output_dir: Directory for output files (default: same as input)

    Returns:
        Analysis results dictionary
    """
    # Load session
    metadata, session_data = load_session(filepath)

    print(f"Session: {metadata.get('session_id', 'unknown')[:8]}")
    print(f"Model: {metadata.get('model', 'unknown')}")
    print(f"Turns: {metadata.get('total_turns', 0)}")
    print(f"Biosignal samples: {metadata.get('biosignal_samples', 0)}")

    # Check versions
    versions = metadata.get('versions', {})
    if metadata.get('versions_inferred'):
        print(f"Versions: INFERRED from date (pre-versioning export)")
    print(f"  core_metrics: {versions.get('core_metrics', 'unknown')}")
    print(f"  extensions: {versions.get('extensions', 'unknown')}")
    print()

    # Extract time series
    ts_data = extract_time_series(session_data)
    if ts_data is None:
        print("ERROR: Could not extract aligned time series")
        return {"error": "No aligned time series data"}

    print(f"Time series: {len(ts_data)} samples @ {ts_data.sample_rate:.2f} Hz")
    print(f"Duration: {ts_data.timestamps[-1]:.1f} seconds")
    print(f"HR range: {ts_data.hr.min():.0f} - {ts_data.hr.max():.0f} bpm")
    print(f"Δκ range: {ts_data.delta_kappa.min():.3f} - {ts_data.delta_kappa.max():.3f}")
    print()

    # Compute cross-correlation
    print(f"Computing cross-correlation (max lag: ±{max_lag_seconds}s, {n_permutations} permutations)...")
    result = compute_cross_correlation(ts_data, max_lag_seconds, n_permutations)

    print("\n=== RESULTS ===")
    print(f"Peak correlation: {result.peak_correlation:.4f}")
    print(f"Peak lag: {result.peak_lag} samples ({result.peak_lag_seconds:.2f} seconds)")
    print(f"95% null threshold: ±{result.null_threshold_95:.4f}")
    print(f"99% null threshold: ±{result.null_threshold_99:.4f}")
    print(f"Significant (p<0.05): {result.is_significant_95}")
    print(f"Significant (p<0.01): {result.is_significant_99}")
    print()
    print("INTERPRETATION:")
    print(result.interpretation)

    # Generate plot if requested
    if plot:
        if output_dir is None:
            output_dir = Path(filepath).parent
        session_id = metadata.get('session_id', 'unknown')[:8]
        plot_path = Path(output_dir) / f"correlogram_{session_id}.png"
        plot_correlogram(result, str(plot_path),
                        title=f"Cross-Correlation: {metadata.get('model', 'unknown')}")

    # Return structured results
    return {
        "session_id": metadata.get('session_id'),
        "model": metadata.get('model'),
        "versions": versions,
        "versions_inferred": metadata.get('versions_inferred', False),
        "n_samples": len(ts_data),
        "duration_seconds": float(ts_data.timestamps[-1]),
        "sample_rate_hz": ts_data.sample_rate,
        "cross_correlation": {
            "peak_correlation": float(result.peak_correlation),
            "peak_lag_samples": int(result.peak_lag),
            "peak_lag_seconds": float(result.peak_lag_seconds),
            "null_threshold_95": float(result.null_threshold_95),
            "null_threshold_99": float(result.null_threshold_99),
            "is_significant_95": result.is_significant_95,
            "is_significant_99": result.is_significant_99,
            "interpretation": result.interpretation
        },
        "analysis_versions": get_versions_dict()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Coupling signature analysis: Cross-correlation between HR and Δκ"
    )
    parser.add_argument("session_file", help="Path to session export JSON")
    parser.add_argument("--max-lag", type=float, default=60.0,
                       help="Maximum lag in seconds (default: 60)")
    parser.add_argument("--permutations", type=int, default=1000,
                       help="Number of permutations for null model (default: 1000)")
    parser.add_argument("--plot", action="store_true",
                       help="Generate correlogram plot")
    parser.add_argument("--output-dir", help="Output directory for plots/results")
    parser.add_argument("--json", action="store_true",
                       help="Output results as JSON")

    args = parser.parse_args()

    results = analyze_session(
        args.session_file,
        max_lag_seconds=args.max_lag,
        n_permutations=args.permutations,
        plot=args.plot,
        output_dir=args.output_dir
    )

    if args.json:
        print("\n=== JSON OUTPUT ===")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
