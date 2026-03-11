"""
Turn-Level Coupling Analysis

Computes turn-level somatic summaries (mean HR averaged over each conversational
turn's duration) and correlates directly with turn-level semantic metrics.

This addresses the interpolation weakness in coupling_analysis.py, which inflates
~16 semantic data points to 1Hz resolution. Here we report honest N.

Two analysis levels:
1. Conversation-turn level: semantic_shift vs mean HR (N ≈ 40)
2. Metrics-snapshot level: Δκ, ΔH, α vs mean HR (N ≈ 16)

Usage:
    python tools/turn_level_coupling.py <session_export.json> [--plot] [--output-dir DIR]
"""

import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats


@dataclass
class TurnSomatic:
    """Somatic summary for a single conversational turn window."""
    turn: int
    start_time: datetime
    end_time: datetime
    duration_s: float
    n_bio_samples: int
    mean_hr: float
    std_hr: float
    min_hr: float
    max_hr: float
    hr_range: float
    # Semantic metrics (from conversation context)
    semantic_shift: Optional[float] = None
    speaker: Optional[str] = None


@dataclass
class MetricsSomatic:
    """Somatic summary matched to a metrics_history snapshot."""
    snapshot_idx: int
    turn_count: int
    timestamp: datetime
    window_start: datetime
    window_end: datetime
    duration_s: float
    n_bio_samples: int
    mean_hr: float
    std_hr: float
    # Semantic metrics (from metrics_history)
    delta_kappa: float
    alpha: float
    delta_h: float
    psi_composite: Optional[float] = None


@dataclass
class CorrelationResult:
    """Result of a single correlation test."""
    metric_name: str
    somatic_name: str
    n: int
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    # Bootstrap CIs
    pearson_ci_low: float
    pearson_ci_high: float
    spearman_ci_low: float
    spearman_ci_high: float


@dataclass
class PhaseComparison:
    """Phase-level comparison of somatic and semantic profiles."""
    phase_name: str
    turn_range: Tuple[int, int]
    n_turns: int
    n_bio_samples: int
    mean_hr: float
    std_hr: float
    mean_semantic_shift: float
    mean_delta_kappa: Optional[float] = None
    mean_alpha: Optional[float] = None
    mean_delta_h: Optional[float] = None


def parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp, handling various formats."""
    ts_str = ts_str.replace('Z', '+00:00')
    # Try with timezone
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        pass
    # Try without timezone
    for fmt in ['%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S']:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts_str}")


def load_session(filepath: str) -> Dict:
    """Load session export JSON."""
    with open(filepath) as f:
        return json.load(f)


def extract_turn_somatics(session: Dict) -> List[TurnSomatic]:
    """
    For each conversation turn, compute somatic summary from biosignal stream.

    Each turn's window runs from its timestamp to the next turn's timestamp.
    The last turn's window runs from its timestamp to the end of biosignal data.
    """
    conversation = session.get('conversation', [])
    biosignal = session.get('biosignal_stream', [])

    if not conversation or not biosignal:
        return []

    # Parse biosignal timestamps and HR
    bio_data = []
    for s in biosignal:
        ts = parse_timestamp(s['ts'])
        bio_data.append((ts, s['hr']))
    bio_data.sort(key=lambda x: x[0])

    # Parse conversation turns
    turns = []
    for entry in conversation:
        ts = parse_timestamp(entry['timestamp'])
        turns.append({
            'turn': entry['turn'],
            'timestamp': ts,
            'semantic_shift': entry.get('context', {}).get('semantic_shift'),
            'speaker': entry.get('speaker', 'unknown'),
        })
    turns.sort(key=lambda x: x['timestamp'])

    # For each turn, find biosignal window
    results = []
    for i, turn in enumerate(turns):
        window_start = turn['timestamp']
        if i + 1 < len(turns):
            window_end = turns[i + 1]['timestamp']
        else:
            # Last turn: window extends to end of biosignal
            window_end = bio_data[-1][0]

        # Find biosignal samples in this window
        hr_values = [hr for ts, hr in bio_data if window_start <= ts < window_end]

        if not hr_values:
            continue

        hr_arr = np.array(hr_values, dtype=float)
        duration = (window_end - window_start).total_seconds()

        results.append(TurnSomatic(
            turn=turn['turn'],
            start_time=window_start,
            end_time=window_end,
            duration_s=duration,
            n_bio_samples=len(hr_values),
            mean_hr=float(np.mean(hr_arr)),
            std_hr=float(np.std(hr_arr)),
            min_hr=float(np.min(hr_arr)),
            max_hr=float(np.max(hr_arr)),
            hr_range=float(np.max(hr_arr) - np.min(hr_arr)),
            semantic_shift=turn['semantic_shift'],
            speaker=turn['speaker'],
        ))

    return results


def extract_metrics_somatics(session: Dict) -> List[MetricsSomatic]:
    """
    For each metrics_history snapshot, compute somatic summary.

    Each snapshot's window runs from its timestamp to the next snapshot's timestamp.
    """
    metrics_history = session.get('metrics_history', [])
    biosignal = session.get('biosignal_stream', [])

    if not metrics_history or not biosignal:
        return []

    # Parse biosignal
    bio_data = []
    for s in biosignal:
        ts = parse_timestamp(s['ts'])
        bio_data.append((ts, s['hr']))
    bio_data.sort(key=lambda x: x[0])

    # Parse metrics snapshots
    snapshots = []
    for entry in metrics_history:
        ts = parse_timestamp(entry['timestamp'])
        metrics = entry.get('metrics', {})
        snapshots.append({
            'timestamp': ts,
            'turn_count': entry.get('turn_count', 0),
            'delta_kappa': metrics.get('delta_kappa', 0),
            'alpha': metrics.get('alpha', 0.5),
            'delta_h': metrics.get('delta_h', 0),
            'psi_composite': entry.get('psi_composite'),
        })
    snapshots.sort(key=lambda x: x['timestamp'])

    results = []
    for i, snap in enumerate(snapshots):
        # Window: from previous snapshot to this one (representing the period leading up to this measurement)
        # Alternative: from this snapshot to next (representing the period this measurement covers)
        # We use: midpoint between previous and this, to midpoint between this and next
        # Actually, simplest and most defensible: from this snapshot to next snapshot
        window_start = snap['timestamp']
        if i + 1 < len(snapshots):
            window_end = snapshots[i + 1]['timestamp']
        else:
            window_end = bio_data[-1][0]

        hr_values = [hr for ts, hr in bio_data if window_start <= ts < window_end]

        if not hr_values:
            continue

        hr_arr = np.array(hr_values, dtype=float)
        duration = (window_end - window_start).total_seconds()

        results.append(MetricsSomatic(
            snapshot_idx=i,
            turn_count=snap['turn_count'],
            timestamp=snap['timestamp'],
            window_start=window_start,
            window_end=window_end,
            duration_s=duration,
            n_bio_samples=len(hr_values),
            mean_hr=float(np.mean(hr_arr)),
            std_hr=float(np.std(hr_arr)),
            delta_kappa=snap['delta_kappa'],
            alpha=snap['alpha'],
            delta_h=snap['delta_h'],
            psi_composite=snap['psi_composite'],
        ))

    return results


def bootstrap_correlation(x: np.ndarray, y: np.ndarray,
                          n_bootstrap: int = 5000,
                          ci: float = 0.95,
                          method: str = 'pearson') -> Tuple[float, float]:
    """
    Bootstrap confidence interval for correlation.

    Returns (ci_low, ci_high).
    """
    n = len(x)
    boot_corrs = []

    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        x_boot = x[idx]
        y_boot = y[idx]

        # Skip if no variance
        if np.std(x_boot) < 1e-10 or np.std(y_boot) < 1e-10:
            continue

        if method == 'pearson':
            r, _ = stats.pearsonr(x_boot, y_boot)
        else:
            r, _ = stats.spearmanr(x_boot, y_boot)

        if np.isfinite(r):
            boot_corrs.append(r)

    if not boot_corrs:
        return (np.nan, np.nan)

    alpha = (1 - ci) / 2
    return (
        float(np.percentile(boot_corrs, 100 * alpha)),
        float(np.percentile(boot_corrs, 100 * (1 - alpha)))
    )


def compute_correlation(x: np.ndarray, y: np.ndarray,
                        x_name: str, y_name: str,
                        n_bootstrap: int = 5000) -> CorrelationResult:
    """Compute Pearson and Spearman correlations with bootstrap CIs."""
    n = len(x)

    # Pearson
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        pearson_r, pearson_p = 0.0, 1.0
        spearman_rho, spearman_p = 0.0, 1.0
    else:
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_rho, spearman_p = stats.spearmanr(x, y)

    # Bootstrap CIs
    p_ci = bootstrap_correlation(x, y, n_bootstrap, method='pearson')
    s_ci = bootstrap_correlation(x, y, n_bootstrap, method='spearman')

    return CorrelationResult(
        metric_name=x_name,
        somatic_name=y_name,
        n=n,
        pearson_r=float(pearson_r),
        pearson_p=float(pearson_p),
        spearman_rho=float(spearman_rho),
        spearman_p=float(spearman_p),
        pearson_ci_low=p_ci[0],
        pearson_ci_high=p_ci[1],
        spearman_ci_low=s_ci[0],
        spearman_ci_high=s_ci[1],
    )


def compute_phase_comparisons(turn_somatics: List[TurnSomatic],
                               metrics_somatics: List[MetricsSomatic],
                               phases: Dict[str, Tuple[int, int]]) -> List[PhaseComparison]:
    """
    Compute phase-level summaries.

    phases: dict mapping phase name to (first_turn, last_turn) inclusive.
    """
    results = []
    for phase_name, (first, last) in phases.items():
        # Filter turn somatics
        phase_turns = [t for t in turn_somatics if first <= t.turn <= last]
        # Filter metrics somatics
        phase_metrics = [m for m in metrics_somatics
                         if first <= m.turn_count <= last]

        if not phase_turns:
            continue

        hr_values = [t.mean_hr for t in phase_turns]
        sem_shifts = [t.semantic_shift for t in phase_turns
                      if t.semantic_shift is not None]
        bio_samples = sum(t.n_bio_samples for t in phase_turns)

        dk = [m.delta_kappa for m in phase_metrics] if phase_metrics else None
        al = [m.alpha for m in phase_metrics] if phase_metrics else None
        dh = [m.delta_h for m in phase_metrics] if phase_metrics else None

        results.append(PhaseComparison(
            phase_name=phase_name,
            turn_range=(first, last),
            n_turns=len(phase_turns),
            n_bio_samples=bio_samples,
            mean_hr=float(np.mean(hr_values)),
            std_hr=float(np.std(hr_values)),
            mean_semantic_shift=float(np.mean(sem_shifts)) if sem_shifts else 0.0,
            mean_delta_kappa=float(np.mean(dk)) if dk else None,
            mean_alpha=float(np.mean(al)) if al else None,
            mean_delta_h=float(np.mean(dh)) if dh else None,
        ))

    return results


def plot_turn_level(turn_somatics: List[TurnSomatic],
                    metrics_somatics: List[MetricsSomatic],
                    correlations: Dict[str, CorrelationResult],
                    output_path: str,
                    session_name: str = "Session"):
    """Generate multi-panel figure for turn-level analysis."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig = plt.figure(figsize=(14, 16))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

    # --- Panel 1: Turn-level HR time series with semantic_shift overlay ---
    ax1 = fig.add_subplot(gs[0, :])
    turns_with_shift = [(t.turn, t.mean_hr, t.semantic_shift)
                        for t in turn_somatics if t.semantic_shift is not None]
    if turns_with_shift:
        t_turns, t_hr, t_shift = zip(*turns_with_shift)
        ax1.plot(t_turns, t_hr, 'o-', color='#c0392b', markersize=4,
                 linewidth=1.5, label='Mean HR (bpm)')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(t_turns, t_shift, 's-', color='#2980b9', markersize=3,
                      linewidth=1, alpha=0.7, label='Semantic shift')
        ax1_twin.set_ylabel('Semantic shift (cosine distance)', color='#2980b9')
        ax1_twin.tick_params(axis='y', labelcolor='#2980b9')
    ax1.set_xlabel('Turn number')
    ax1.set_ylabel('Mean HR (bpm)', color='#c0392b')
    ax1.tick_params(axis='y', labelcolor='#c0392b')
    ax1.set_title(f'{session_name} — Turn-Level HR and Semantic Shift')
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Scatter — semantic_shift vs mean HR ---
    ax2 = fig.add_subplot(gs[1, 0])
    turns_valid = [(t.semantic_shift, t.mean_hr) for t in turn_somatics
                   if t.semantic_shift is not None]
    if turns_valid:
        shifts, hrs = zip(*turns_valid)
        ax2.scatter(shifts, hrs, alpha=0.6, color='#8e44ad', s=30)
        # Trend line
        if len(shifts) > 2:
            z = np.polyfit(shifts, hrs, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(shifts), max(shifts), 50)
            ax2.plot(x_range, p(x_range), '--', color='gray', alpha=0.5)
    corr = correlations.get('semantic_shift_vs_hr')
    if corr:
        ax2.set_title(f'Semantic Shift vs HR\nr={corr.pearson_r:.3f} '
                       f'[{corr.pearson_ci_low:.3f}, {corr.pearson_ci_high:.3f}]\n'
                       f'N={corr.n}')
    else:
        ax2.set_title('Semantic Shift vs HR')
    ax2.set_xlabel('Semantic shift')
    ax2.set_ylabel('Mean HR (bpm)')
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Scatter — Δκ vs mean HR ---
    ax3 = fig.add_subplot(gs[1, 1])
    if metrics_somatics:
        dk_vals = [m.delta_kappa for m in metrics_somatics]
        hr_vals = [m.mean_hr for m in metrics_somatics]
        ax3.scatter(dk_vals, hr_vals, alpha=0.7, color='#e67e22', s=40,
                    edgecolors='black', linewidth=0.5)
        if len(dk_vals) > 2:
            z = np.polyfit(dk_vals, hr_vals, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(dk_vals), max(dk_vals), 50)
            ax3.plot(x_range, p(x_range), '--', color='gray', alpha=0.5)
    corr = correlations.get('delta_kappa_vs_hr')
    if corr:
        ax3.set_title(f'Δκ vs HR (metrics snapshots)\nr={corr.pearson_r:.3f} '
                       f'[{corr.pearson_ci_low:.3f}, {corr.pearson_ci_high:.3f}]\n'
                       f'N={corr.n}')
    ax3.set_xlabel('Δκ (semantic curvature)')
    ax3.set_ylabel('Mean HR (bpm)')
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Scatter — α vs mean HR ---
    ax4 = fig.add_subplot(gs[2, 0])
    if metrics_somatics:
        a_vals = [m.alpha for m in metrics_somatics]
        hr_vals = [m.mean_hr for m in metrics_somatics]
        ax4.scatter(a_vals, hr_vals, alpha=0.7, color='#27ae60', s=40,
                    edgecolors='black', linewidth=0.5)
        if len(a_vals) > 2:
            z = np.polyfit(a_vals, hr_vals, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(a_vals), max(a_vals), 50)
            ax4.plot(x_range, p(x_range), '--', color='gray', alpha=0.5)
    corr = correlations.get('alpha_vs_hr')
    if corr:
        ax4.set_title(f'α vs HR (metrics snapshots)\nr={corr.pearson_r:.3f} '
                       f'[{corr.pearson_ci_low:.3f}, {corr.pearson_ci_high:.3f}]\n'
                       f'N={corr.n}')
    ax4.set_xlabel('α (fractal similarity)')
    ax4.set_ylabel('Mean HR (bpm)')
    ax4.grid(True, alpha=0.3)

    # --- Panel 5: Scatter — ΔH vs mean HR ---
    ax5 = fig.add_subplot(gs[2, 1])
    if metrics_somatics:
        dh_vals = [m.delta_h for m in metrics_somatics]
        hr_vals = [m.mean_hr for m in metrics_somatics]
        ax5.scatter(dh_vals, hr_vals, alpha=0.7, color='#2c3e50', s=40,
                    edgecolors='black', linewidth=0.5)
        if len(dh_vals) > 2 and np.std(dh_vals) > 1e-10:
            z = np.polyfit(dh_vals, hr_vals, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(dh_vals), max(dh_vals), 50)
            ax5.plot(x_range, p(x_range), '--', color='gray', alpha=0.5)
    corr = correlations.get('delta_h_vs_hr')
    if corr:
        ax5.set_title(f'ΔH vs HR (metrics snapshots)\nr={corr.pearson_r:.3f} '
                       f'[{corr.pearson_ci_low:.3f}, {corr.pearson_ci_high:.3f}]\n'
                       f'N={corr.n}')
    ax5.set_xlabel('ΔH (entropy shift)')
    ax5.set_ylabel('Mean HR (bpm)')
    ax5.grid(True, alpha=0.3)

    # --- Panel 6: Metrics-snapshot time series ---
    ax6 = fig.add_subplot(gs[3, :])
    if metrics_somatics:
        snap_turns = [m.turn_count for m in metrics_somatics]
        snap_hr = [m.mean_hr for m in metrics_somatics]
        snap_dk = [m.delta_kappa for m in metrics_somatics]
        snap_a = [m.alpha for m in metrics_somatics]

        ax6.plot(snap_turns, snap_hr, 'o-', color='#c0392b', markersize=5,
                 linewidth=1.5, label='Mean HR')
        ax6_twin = ax6.twinx()
        ax6_twin.plot(snap_turns, snap_dk, '^-', color='#e67e22', markersize=4,
                      linewidth=1, alpha=0.7, label='Δκ')
        ax6_twin.plot(snap_turns, snap_a, 'v-', color='#27ae60', markersize=4,
                      linewidth=1, alpha=0.7, label='α')
        ax6_twin.set_ylabel('Semantic metrics')
        ax6_twin.legend(loc='upper right')
    ax6.set_xlabel('Turn count (at metrics snapshot)')
    ax6.set_ylabel('Mean HR (bpm)', color='#c0392b')
    ax6.tick_params(axis='y', labelcolor='#c0392b')
    ax6.set_title('Metrics Snapshots — HR with Δκ and α')
    ax6.legend(loc='upper left')
    ax6.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {output_path}")
    plt.close()


def analyze_session(filepath: str, plot: bool = False,
                    output_dir: Optional[str] = None) -> Dict:
    """Run full turn-level coupling analysis."""
    session = load_session(filepath)
    metadata = session.get('metadata', {})

    session_name = metadata.get('model', 'Unknown')
    total_turns = metadata.get('total_turns', 0)
    bio_samples = metadata.get('biosignal_samples', 0)

    print(f"=" * 60)
    print(f"TURN-LEVEL COUPLING ANALYSIS")
    print(f"=" * 60)
    print(f"Session: {metadata.get('session_id', 'unknown')[:8]}")
    print(f"Model: {session_name}")
    print(f"Total turns: {total_turns}")
    print(f"Biosignal samples: {bio_samples}")
    print()

    # === Extract turn-level somatics ===
    turn_somatics = extract_turn_somatics(session)
    print(f"Turn-level somatics extracted: {len(turn_somatics)} turns")
    if turn_somatics:
        durations = [t.duration_s for t in turn_somatics]
        samples = [t.n_bio_samples for t in turn_somatics]
        print(f"  Turn durations: {np.min(durations):.1f}s – {np.max(durations):.1f}s "
              f"(median {np.median(durations):.1f}s)")
        print(f"  Bio samples per turn: {np.min(samples)} – {np.max(samples)} "
              f"(median {np.median(samples):.0f})")
        print(f"  HR range across turns: {min(t.mean_hr for t in turn_somatics):.1f} – "
              f"{max(t.mean_hr for t in turn_somatics):.1f} bpm")
    print()

    # === Extract metrics-level somatics ===
    metrics_somatics = extract_metrics_somatics(session)
    print(f"Metrics-level somatics extracted: {len(metrics_somatics)} snapshots")
    if metrics_somatics:
        print(f"  Turn counts: {[m.turn_count for m in metrics_somatics]}")
        print(f"  HR range: {min(m.mean_hr for m in metrics_somatics):.1f} – "
              f"{max(m.mean_hr for m in metrics_somatics):.1f} bpm")
        print(f"  Δκ range: {min(m.delta_kappa for m in metrics_somatics):.3f} – "
              f"{max(m.delta_kappa for m in metrics_somatics):.3f}")
        print(f"  α range: {min(m.alpha for m in metrics_somatics):.3f} – "
              f"{max(m.alpha for m in metrics_somatics):.3f}")
        print(f"  ΔH range: {min(m.delta_h for m in metrics_somatics):.3f} – "
              f"{max(m.delta_h for m in metrics_somatics):.3f}")
    print()

    # === Correlations ===
    correlations = {}

    # 1. Semantic shift vs HR (conversation-turn level)
    print(f"-" * 60)
    print(f"ANALYSIS 1: Conversation-Turn Level (semantic_shift vs HR)")
    print(f"-" * 60)

    valid_turns = [(t.semantic_shift, t.mean_hr) for t in turn_somatics
                   if t.semantic_shift is not None]
    if len(valid_turns) >= 5:
        shifts, hrs = zip(*valid_turns)
        shifts_arr = np.array(shifts)
        hrs_arr = np.array(hrs)

        corr = compute_correlation(shifts_arr, hrs_arr,
                                   'semantic_shift', 'mean_HR')
        correlations['semantic_shift_vs_hr'] = corr

        print(f"  N = {corr.n}")
        print(f"  Pearson r = {corr.pearson_r:.4f}  p = {corr.pearson_p:.4f}  "
              f"95% CI [{corr.pearson_ci_low:.4f}, {corr.pearson_ci_high:.4f}]")
        print(f"  Spearman ρ = {corr.spearman_rho:.4f}  p = {corr.spearman_p:.4f}  "
              f"95% CI [{corr.spearman_ci_low:.4f}, {corr.spearman_ci_high:.4f}]")
    else:
        print(f"  Insufficient data points ({len(valid_turns)})")
    print()

    # 2. Metrics snapshots vs HR
    print(f"-" * 60)
    print(f"ANALYSIS 2: Metrics-Snapshot Level (Δκ, α, ΔH vs HR)")
    print(f"-" * 60)

    if len(metrics_somatics) >= 5:
        dk_arr = np.array([m.delta_kappa for m in metrics_somatics])
        a_arr = np.array([m.alpha for m in metrics_somatics])
        dh_arr = np.array([m.delta_h for m in metrics_somatics])
        hr_arr = np.array([m.mean_hr for m in metrics_somatics])

        for name, vals in [('delta_kappa', dk_arr), ('alpha', a_arr), ('delta_h', dh_arr)]:
            if np.std(vals) < 1e-10:
                print(f"  {name}: No variance (all values ≈ {vals[0]:.3f}), skipping")
                continue
            corr = compute_correlation(vals, hr_arr, name, 'mean_HR')
            correlations[f'{name}_vs_hr'] = corr
            print(f"  {name} vs HR:")
            print(f"    N = {corr.n}")
            print(f"    Pearson r = {corr.pearson_r:.4f}  p = {corr.pearson_p:.4f}  "
                  f"95% CI [{corr.pearson_ci_low:.4f}, {corr.pearson_ci_high:.4f}]")
            print(f"    Spearman ρ = {corr.spearman_rho:.4f}  p = {corr.spearman_p:.4f}  "
                  f"95% CI [{corr.spearman_ci_low:.4f}, {corr.spearman_ci_high:.4f}]")

        # Also Ψ composite if available
        psi_vals = [m.psi_composite for m in metrics_somatics if m.psi_composite is not None]
        if len(psi_vals) == len(metrics_somatics):
            psi_arr = np.array(psi_vals)
            corr = compute_correlation(psi_arr, hr_arr, 'psi_composite', 'mean_HR')
            correlations['psi_composite_vs_hr'] = corr
            print(f"  psi_composite vs HR:")
            print(f"    N = {corr.n}")
            print(f"    Pearson r = {corr.pearson_r:.4f}  p = {corr.pearson_p:.4f}  "
                  f"95% CI [{corr.pearson_ci_low:.4f}, {corr.pearson_ci_high:.4f}]")
            print(f"    Spearman ρ = {corr.spearman_rho:.4f}  p = {corr.spearman_p:.4f}  "
                  f"95% CI [{corr.spearman_ci_low:.4f}, {corr.spearman_ci_high:.4f}]")
    else:
        print(f"  Insufficient metrics snapshots ({len(metrics_somatics)})")
    print()

    # === Phase-level comparison ===
    print(f"-" * 60)
    print(f"ANALYSIS 3: Phase-Level Comparison")
    print(f"-" * 60)

    # Grunch session phases (from analysis doc)
    grunch_phases = {
        'Setup & Surface': (1, 10),
        'Jungian Depth': (11, 18),
        'Primal & Hegemonic': (19, 24),
        'Grunch Denial': (25, 40),
    }

    phase_results = compute_phase_comparisons(turn_somatics, metrics_somatics, grunch_phases)

    if phase_results:
        print(f"\n  {'Phase':<25} {'Turns':<8} {'N bio':<8} {'Mean HR':<10} "
              f"{'Sem. Shift':<12} {'Δκ':<8} {'α':<8} {'ΔH':<8}")
        print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*10} {'─'*12} {'─'*8} {'─'*8} {'─'*8}")
        for p in phase_results:
            dk_str = f"{p.mean_delta_kappa:.3f}" if p.mean_delta_kappa is not None else "—"
            a_str = f"{p.mean_alpha:.3f}" if p.mean_alpha is not None else "—"
            dh_str = f"{p.mean_delta_h:.3f}" if p.mean_delta_h is not None else "—"
            print(f"  {p.phase_name:<25} {p.n_turns:<8} {p.n_bio_samples:<8} "
                  f"{p.mean_hr:<10.1f} {p.mean_semantic_shift:<12.3f} "
                  f"{dk_str:<8} {a_str:<8} {dh_str:<8}")
    print()

    # === Summary ===
    print(f"=" * 60)
    print(f"SUMMARY")
    print(f"=" * 60)

    sig_count = 0
    for name, corr in correlations.items():
        sig = "**" if corr.pearson_p < 0.01 else "*" if corr.pearson_p < 0.05 else ""
        if corr.pearson_p < 0.05:
            sig_count += 1
        print(f"  {name}: r={corr.pearson_r:.3f} {sig}  (N={corr.n})")

    print(f"\n  {sig_count}/{len(correlations)} correlations significant at p<0.05")
    print(f"  Note: N={len(metrics_somatics)} for metrics-level, "
          f"N={len(valid_turns)} for turn-level")
    print(f"  These are honest sample sizes — no interpolation inflation.")

    # === Plot ===
    if plot:
        if output_dir is None:
            output_dir = Path(filepath).parent
        session_id = metadata.get('session_id', 'unknown')[:8]
        plot_path = Path(output_dir) / f"turn_level_{session_id}.png"
        plot_turn_level(turn_somatics, metrics_somatics, correlations,
                        str(plot_path), session_name)

    # === Return structured results ===
    return {
        'session_id': metadata.get('session_id'),
        'model': metadata.get('model'),
        'n_conversation_turns': total_turns,
        'n_metrics_snapshots': len(metrics_somatics),
        'n_biosignal_samples': bio_samples,
        'turn_level': {
            'n': len(valid_turns),
            'correlation': {
                'metric': 'semantic_shift',
                **({k: getattr(correlations['semantic_shift_vs_hr'], k)
                    for k in ['pearson_r', 'pearson_p', 'pearson_ci_low', 'pearson_ci_high',
                              'spearman_rho', 'spearman_p', 'spearman_ci_low', 'spearman_ci_high']}
                   if 'semantic_shift_vs_hr' in correlations else {})
            }
        },
        'metrics_level': {
            'n': len(metrics_somatics),
            'correlations': {
                name.replace('_vs_hr', ''): {
                    k: getattr(corr, k)
                    for k in ['pearson_r', 'pearson_p', 'pearson_ci_low', 'pearson_ci_high',
                              'spearman_rho', 'spearman_p', 'spearman_ci_low', 'spearman_ci_high']
                }
                for name, corr in correlations.items() if name != 'semantic_shift_vs_hr'
            }
        },
        'phase_comparisons': [
            {
                'phase': p.phase_name,
                'turn_range': list(p.turn_range),
                'n_turns': p.n_turns,
                'mean_hr': p.mean_hr,
                'mean_semantic_shift': p.mean_semantic_shift,
                'mean_delta_kappa': p.mean_delta_kappa,
                'mean_alpha': p.mean_alpha,
                'mean_delta_h': p.mean_delta_h,
            }
            for p in phase_results
        ],
        'turn_somatics': [
            {
                'turn': t.turn,
                'speaker': t.speaker,
                'duration_s': t.duration_s,
                'n_bio_samples': t.n_bio_samples,
                'mean_hr': t.mean_hr,
                'semantic_shift': t.semantic_shift,
            }
            for t in turn_somatics
        ],
        'metrics_somatics': [
            {
                'turn_count': m.turn_count,
                'n_bio_samples': m.n_bio_samples,
                'mean_hr': m.mean_hr,
                'delta_kappa': m.delta_kappa,
                'alpha': m.alpha,
                'delta_h': m.delta_h,
                'psi_composite': m.psi_composite,
            }
            for m in metrics_somatics
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Turn-level coupling analysis: honest N, no interpolation"
    )
    parser.add_argument("session_file", help="Path to session export JSON")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--json", action="store_true", help="Output JSON results")

    args = parser.parse_args()

    results = analyze_session(
        args.session_file,
        plot=args.plot,
        output_dir=args.output_dir
    )

    if args.json:
        print("\n=== JSON OUTPUT ===")
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
