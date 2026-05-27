"""
Protocol Comparison Analysis Script

This script analyzes differences in negotiation outcomes across two protocols
(Traditional vs LLM).

It extracts final agreement data from session log files and generates:
- Descriptive statistics
- Comparative visualizations (box plots, bar charts)
- Excel workbook with multiple analysis sheets
"""

import os
import sys
import glob
import functools
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from significance_tests import perform_pairwise_comparison, interpret_effect_size

PROTOCOL_ORDER = ['LLM', 'Traditional']
# Map raw filename tokens -> canonical protocol labels used everywhere downstream.
FILENAME_PROTOCOL_MAP = {'TRADITIONAL': 'Traditional', 'LLM': 'LLM'}
# Canonical protocol colors — used across every figure for consistency.
PROTOCOL_COLORS = {
    'LLM': '#3B7FB6',  # blue
    'Traditional':     '#E07B3A',  # orange
}

# Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SESSION_LOGS_DIR = os.path.join(PROJECT_ROOT, "data", "session_logs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "evaluation", "emperical")

METRICS = ['HumanUtility', 'AgentUtility', 'NashDistance', 'ProductScore']
SIDE_METRICS = ['Round', 'Time']
LOWER_IS_BETTER = {'NashDistance', 'Round', 'Time'}


def summarize_series(values: pd.Series) -> Tuple[float, float]:
    clean_values = values.dropna()
    if clean_values.empty:
        return np.nan, np.nan
    return float(np.mean(clean_values)), float(np.std(clean_values))


def parse_session_filename(filename: str) -> Dict[str, str]:
    stem = filename.removesuffix('.xlsx')
    parts = stem.split('_')

    if len(parts) >= 6 and all(part.isdigit() for part in parts[-3:]):
        timestamp_parts = parts[-3:]
        raw_protocol = parts[-4].upper()
        domain = parts[-5].upper()
        user_id = '_'.join(parts[:-5])
    elif len(parts) >= 5 and parts[-1].isdigit() and parts[-2].isdigit():
        timestamp_parts = parts[-2:]
        raw_protocol = parts[-3].upper()
        domain = parts[-4].upper()
        user_id = '_'.join(parts[:-4])
    else:
        raise ValueError(f"Invalid filename format: {filename}")

    if raw_protocol not in FILENAME_PROTOCOL_MAP:
        raise ValueError(f"Invalid protocol '{raw_protocol}' in file: {filename}")

    if not user_id:
        raise ValueError(f"Missing user ID in file: {filename}")

    return {
        'UserID': user_id,
        'Domain': domain,
        'Protocol': FILENAME_PROTOCOL_MAP[raw_protocol],
        'Timestamp': '_'.join(timestamp_parts)
    }


def build_paired_protocol_data(df: pd.DataFrame, metric: str) -> Optional[pd.DataFrame]:
    protocol_frames = {}

    for protocol in PROTOCOL_ORDER:
        protocol_df = df[df['Protocol'] == protocol][['UserID', metric]].rename(
            columns={metric: protocol}
        )
        protocol_frames[protocol] = protocol_df

    paired_df = protocol_frames['Traditional'].merge(
        protocol_frames['LLM'],
        on='UserID',
        how='inner'
    ).dropna()

    return paired_df if not paired_df.empty else None


def load_session_files(directory: str) -> pd.DataFrame:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Session logs directory not found: {directory}")

    file_pattern = os.path.join(directory, "*.xlsx")
    files = [
        file_path
        for file_path in glob.glob(file_pattern)
        if not os.path.basename(file_path).startswith('~$')
    ]

    if len(files) == 0:
        raise FileNotFoundError(f"No .xlsx files found in {directory}")

    print(f"Found {len(files)} session log files")

    nash_h, nash_a = _canonical_nash_point()

    def nash_dist(h: float, a: float) -> float:
        return float(np.sqrt((h - nash_h) ** 2 + (a - nash_a) ** 2))

    records = []
    skipped_empty = 0

    for file_path in files:
        filename = os.path.basename(file_path)
        metadata = parse_session_filename(filename)

        df = pd.read_excel(file_path, sheet_name='Session')
        if len(df) == 0:
            skipped_empty += 1
            continue

        final_row = df.iloc[-1]
        round_count = float(final_row['Round'])
        elapsed_time = float(final_row['ElapsedTime'])

        if final_row['Action'] == 'Accept':
            h = float(final_row['AgentAUtility'])
            a = float(final_row['AgentBUtility'])
            records.append({
                'Filename': filename,
                **metadata,
                'Status': 'Success',
                'HumanUtility': h,
                'AgentUtility': a,
                'NashDistance': nash_dist(h, a),
                'ProductScore': h * a,
                'Round': round_count,
                'Time': elapsed_time,
            })
        else:
            records.append({
                'Filename': filename,
                **metadata,
                'Status': 'Failed',
                'HumanUtility': 0.0,
                'AgentUtility': 0.0,
                'NashDistance': nash_dist(0.0, 0.0),
                'ProductScore': 0.0,
                'Round': round_count,
                'Time': elapsed_time,
            })

    if not records:
        raise ValueError(f"No session files could be processed from {directory}")

    result_df = pd.DataFrame(records)

    # Drop duplicates: keep only the latest timestamp per (UserID, Protocol).
    # Some users (e.g. melisa) have re-run sessions that should not be
    # double-counted.
    before = len(result_df)
    result_df = (
        result_df.sort_values("Timestamp")
        .drop_duplicates(subset=["UserID", "Protocol"], keep="last")
        .reset_index(drop=True)
    )
    deduped = before - len(result_df)

    print(f"Loaded {len(result_df)} sessions from {len(files)} files")
    print(f"  - Skipped {skipped_empty} empty session files")
    print(f"  - Dropped {deduped} duplicate (UserID, Protocol) sessions (kept latest)")
    print(f"  - Failed (disagreement): {len(result_df[result_df['Status'] == 'Failed'])}")

    return result_df


def pair_on_protocol(df: pd.DataFrame, label: str) -> pd.DataFrame:
    users_per_protocol = df.groupby('UserID')['Protocol'].nunique()
    paired_users = users_per_protocol[users_per_protocol == len(PROTOCOL_ORDER)].index
    dropped_unpaired = df[~df['UserID'].isin(paired_users)]
    paired_df = df[df['UserID'].isin(paired_users)].reset_index(drop=True)

    print(f"[{label}] {len(paired_df)} sessions from {paired_df['UserID'].nunique()} paired users "
          f"(dropped {len(dropped_unpaired)} from {dropped_unpaired['UserID'].nunique()} unpaired)")
    print(f"  - Protocols: {paired_df['Protocol'].value_counts().to_dict()}")

    return paired_df


def compute_descriptive_stats(df: pd.DataFrame, group_cols: List[str],
                              metrics: List[str]) -> pd.DataFrame:
    """
    Compute descriptive statistics for metrics grouped by specified columns.

    Args:
        df: DataFrame with session data
        group_cols: Columns to group by (e.g., ['Protocol'] or ['Domain', 'Protocol'])
        metrics: List of metric column names to analyze

    Returns:
        DataFrame with statistics: Count, Mean, Median, Std, Min, Max, Q1, Q3
    """
    stats_list = []

    for group_values, group_df in df.groupby(group_cols):
        # Handle single group column vs multiple
        if len(group_cols) == 1:
            group_dict = {group_cols[0]: group_values}
        else:
            group_dict = dict(zip(group_cols, group_values))

        for metric in metrics:
            values = group_df[metric].dropna()

            if len(values) == 0:
                continue

            stats = {
                **group_dict,
                'Metric': metric,
                'Count': len(values),
                'Mean': np.mean(values),
                'Median': np.median(values),
                'Std': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values),
                'Q1': np.percentile(values, 25),
                'Q3': np.percentile(values, 75)
            }

            stats_list.append(stats)

    return pd.DataFrame(stats_list)


def compute_success_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute success rates for different groupings.

    Args:
        df: DataFrame with session data

    Returns:
        DataFrame with success rate analysis
    """
    results = []

    # Overall
    total = len(df)
    success_count = len(df[df['Status'] == 'Success'])
    failed_count = len(df[df['Status'] == 'Failed'])

    results.append({
        'Group': 'Overall',
        'Protocol': 'All',
        'Total_Sessions': total,
        'Success_Count': success_count,
        'Failed_Count': failed_count,
        'Success_Rate': (success_count / total * 100) if total > 0 else 0
    })

    # By Protocol (aggregated across domains)
    for protocol in df['Protocol'].unique():
        protocol_df = df[df['Protocol'] == protocol]
        total = len(protocol_df)
        success_count = len(protocol_df[protocol_df['Status'] == 'Success'])
        failed_count = len(protocol_df[protocol_df['Status'] == 'Failed'])

        results.append({
            'Group': f'{protocol}',
            'Protocol': protocol,
            'Total_Sessions': total,
            'Success_Count': success_count,
            'Failed_Count': failed_count,
            'Success_Rate': (success_count / total * 100) if total > 0 else 0
        })

    return pd.DataFrame(results)


def create_paired_violins(df: pd.DataFrame, output_path: str):
    """
    Per-metric paired comparison: half-violin + box + jittered dots +
    connecting lines for successful agreements. A thin broken-axis strip
    at the end (bottom for utilities, top for Nash distance) shows users
    whose sessions ended in disagreement — keeping disagreements visible
    without the 0-imputed cluster distorting the main distribution.
    """
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    sns.set_theme(style='ticks', context='notebook', font='DejaVu Sans')

    flat_color = '#B0B0B0'
    box_color = '#2B2B2B'
    disagree_color = '#888888'
    protocols = ['Traditional', 'LLM']
    protocol_colors = [PROTOCOL_COLORS[p] for p in protocols]
    positions = [1.0, 2.0]
    dot_jitter = 0.045
    rng = np.random.default_rng(42)

    nash_h, nash_a = _canonical_nash_point()
    nash_dis_val = float(np.sqrt(nash_h ** 2 + nash_a ** 2))

    # (metric, ylabel, row, col, dis_side, dis_value)
    metric_info = [
        ('HumanUtility', 'Human Utility', 0, 0, 'bottom', 0.0),
        ('AgentUtility', 'Agent Utility', 0, 1, 'bottom', 0.0),
        ('NashDistance', 'Nash Distance', 1, 0, 'top', nash_dis_val),
        ('ProductScore', 'Product Score (H × A)', 1, 1, 'bottom', 0.0),
    ]

    fig = plt.figure(figsize=(13, 11))
    outer = GridSpec(2, 2, figure=fig, wspace=0.22, hspace=0.30)

    for metric, ylabel, row, col, dis_side, dis_value in metric_info:
        paired = build_paired_protocol_data(df, metric)
        if paired is None or len(paired) == 0:
            continue

        y_alt_all = paired['Traditional'].values.astype(float)
        y_part_all = paired['LLM'].values.astype(float)

        test_name, p_value, effect_size = perform_pairwise_comparison(
            y_alt_all.tolist(), y_part_all.tolist()
        )
        if p_value < 0.001:
            stars = '***'
        elif p_value < 0.01:
            stars = '**'
        elif p_value < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'

        if dis_value is not None:
            is_succ = (y_alt_all != dis_value) & (y_part_all != dis_value)
        else:
            is_succ = np.ones(len(y_alt_all), dtype=bool)

        y_alt = y_alt_all[is_succ]
        y_part = y_part_all[is_succ]
        n_succ = len(y_alt)

        use_break = dis_side is not None and (~is_succ).any()
        if use_break:
            height_ratios = [1, 6] if dis_side == 'top' else [6, 1]
            inner = outer[row, col].subgridspec(2, 1, height_ratios=height_ratios, hspace=0.10)
            ax_strip = fig.add_subplot(inner[0] if dis_side == 'top' else inner[1])
            ax_main = fig.add_subplot(inner[1] if dis_side == 'top' else inner[0])
        else:
            ax_main = fig.add_subplot(outer[row, col])
            ax_strip = None

        if n_succ >= 2:
            parts = ax_main.violinplot([y_alt, y_part], positions=positions, widths=0.9,
                                       showmeans=False, showmedians=False, showextrema=False)
            for body, protocol, pos in zip(parts['bodies'], protocols, positions):
                verts = body.get_paths()[0].vertices
                if protocol == 'Traditional':
                    verts[:, 0] = np.clip(verts[:, 0], -np.inf, pos)
                else:
                    verts[:, 0] = np.clip(verts[:, 0], pos, np.inf)
                body.set_facecolor(PROTOCOL_COLORS[protocol])
                body.set_alpha(0.35)
                body.set_edgecolor('none')

        delta = y_part - y_alt
        posa_better = delta < 0 if metric in LOWER_IS_BETTER else delta > 0
        complete_better = delta > 0 if metric in LOWER_IS_BETTER else delta < 0
        line_colors = np.where(posa_better, PROTOCOL_COLORS['LLM'],
                       np.where(complete_better, PROTOCOL_COLORS['Traditional'], flat_color))

        jitter_alt = rng.uniform(-dot_jitter, dot_jitter, n_succ)
        jitter_part = rng.uniform(-dot_jitter, dot_jitter, n_succ)
        x_alt = positions[0] + jitter_alt
        x_part = positions[1] + jitter_part

        for i in range(n_succ):
            ax_main.plot([x_alt[i], x_part[i]], [y_alt[i], y_part[i]],
                         color=line_colors[i], alpha=0.28, linewidth=0.7, zorder=2,
                         solid_capstyle='round')

        ax_main.scatter(x_alt, y_alt, c=line_colors, s=16, alpha=0.85,
                        edgecolor='white', linewidth=0.4, zorder=4)
        ax_main.scatter(x_part, y_part, c=line_colors, s=16, alpha=0.85,
                        edgecolor='white', linewidth=0.4, zorder=4)

        if n_succ >= 2:
            bp = ax_main.boxplot([y_alt, y_part], positions=positions, widths=0.18,
                                 patch_artist=True, showfliers=False, zorder=3,
                                 medianprops=dict(color=box_color, linewidth=2.2),
                                 whiskerprops=dict(color=box_color, linewidth=1.2),
                                 capprops=dict(color=box_color, linewidth=1.2))
            for patch in bp['boxes']:
                patch.set_facecolor('white')
                patch.set_edgecolor(box_color)
                patch.set_linewidth(1.3)

        if n_succ >= 2:
            y_hi = max(y_alt.max(), y_part.max())
            y_lo = min(y_alt.min(), y_part.min())
            span = max(y_hi - y_lo, 1e-6)
            bracket_y = y_hi + span * 0.10
            tick_y = bracket_y - span * 0.02
            ax_main.plot([positions[0], positions[0], positions[1], positions[1]],
                         [tick_y, bracket_y, bracket_y, tick_y],
                         color=box_color, linewidth=1.2, zorder=7)
            ax_main.text(sum(positions) / 2, bracket_y + span * 0.015, stars,
                         ha='center', va='bottom', fontsize=14, fontweight='bold',
                         color=box_color)

            break_pad = span * 0.18
            if dis_side == 'bottom':
                ax_main.set_ylim(y_lo - break_pad, bracket_y + span * 0.14)
            elif dis_side == 'top':
                ax_main.set_ylim(y_lo - span * 0.05, bracket_y + break_pad)
            else:
                ax_main.set_ylim(y_lo - span * 0.05, bracket_y + span * 0.14)

        d_ax = ax_strip if (ax_strip is not None and dis_side == 'top') else ax_main
        d_ax.text(0.98, 0.98, f'd = {abs(effect_size):.2f}',
                  transform=d_ax.transAxes, ha='right', va='top',
                  fontsize=11, color='#555555')

        if ax_strip is not None:
            from matplotlib.patches import ConnectionPatch

            fail_idx = np.where(~is_succ)[0]
            for idx in fail_idx:
                a_val = y_alt_all[idx]
                p_val = y_part_all[idx]
                a_fail = (a_val == dis_value)
                p_fail = (p_val == dis_value)

                d_sign = p_val - a_val
                if metric in LOWER_IS_BETTER:
                    if d_sign < 0:
                        lc = PROTOCOL_COLORS['LLM']
                    elif d_sign > 0:
                        lc = PROTOCOL_COLORS['Traditional']
                    else:
                        lc = flat_color
                else:
                    if d_sign > 0:
                        lc = PROTOCOL_COLORS['LLM']
                    elif d_sign < 0:
                        lc = PROTOCOL_COLORS['Traditional']
                    else:
                        lc = flat_color

                ja = float(rng.uniform(-dot_jitter, dot_jitter))
                jp = float(rng.uniform(-dot_jitter, dot_jitter))
                xa = positions[0] + ja
                xp = positions[1] + jp

                ax_a = ax_strip if a_fail else ax_main
                ax_p = ax_strip if p_fail else ax_main

                if lc is not flat_color:
                    if ax_a is ax_p:
                        ax_a.plot([xa, xp], [a_val, p_val], color=lc, alpha=0.30,
                                  linewidth=0.7, zorder=2, solid_capstyle='round')
                    else:
                        con = ConnectionPatch(
                            xyA=(xa, a_val), coordsA=ax_a.transData,
                            xyB=(xp, p_val), coordsB=ax_p.transData,
                            color=lc, alpha=0.30, linewidth=0.7, zorder=2,
                        )
                        fig.add_artist(con)

                ax_a.scatter([xa], [a_val], s=18, alpha=0.85, color=lc,
                             edgecolor='white', linewidth=0.4, zorder=4)
                ax_p.scatter([xp], [p_val], s=18, alpha=0.85, color=lc,
                             edgecolor='white', linewidth=0.4, zorder=4)

            strip_pad = max(span, 0.02) * 0.10
            strip_break_pad = max(span, 0.02) * 0.10
            if dis_side == 'bottom':
                ax_strip.set_ylim(dis_value - strip_pad,
                                  dis_value + strip_break_pad)
            else:
                ax_strip.set_ylim(dis_value - strip_break_pad,
                                  dis_value + strip_pad)
            ax_strip.set_yticks([dis_value])
            if col == 0:
                ax_strip.set_yticklabels(['Disagreements'])
            else:
                ax_strip.set_yticklabels([])
            ax_strip.tick_params(axis='y', labelsize=11)

        for a in [ax_main, ax_strip]:
            if a is None:
                continue
            a.set_xticks(positions)
            a.set_xlim(0.35, 2.65)

        bottom_ax = ax_strip if (ax_strip is not None and dis_side == 'bottom') else ax_main
        bottom_ax.set_xticklabels(protocols, fontsize=14, fontweight='bold')
        for a in [ax_main, ax_strip]:
            if a is None or a is bottom_ax:
                continue
            a.set_xticklabels([])

        ax_main.set_ylabel(ylabel, fontsize=15, fontweight='bold')
        ax_main.tick_params(axis='both', labelsize=13)

        sns.despine(ax=ax_main, trim=False)
        if ax_strip is not None:
            sns.despine(ax=ax_strip, trim=False)
            if dis_side == 'bottom':
                ax_main.spines['bottom'].set_visible(False)
                ax_strip.spines['top'].set_visible(False)
                ax_main.tick_params(bottom=False, labelbottom=False)
            else:
                ax_main.spines['top'].set_visible(False)
                ax_strip.spines['bottom'].set_visible(False)
                ax_strip.tick_params(bottom=False, labelbottom=False)

            dx = 0.022
            gap = 0.6
            kwargs = dict(color=box_color, clip_on=False, linewidth=1.8,
                          transform=fig.transFigure)
            bbox_main = ax_main.get_position()
            bbox_strip = ax_strip.get_position()
            x0 = bbox_main.x0 - dx * bbox_main.width
            x1 = bbox_main.x0 + dx * bbox_main.width
            if dis_side == 'bottom':
                y_mid_hi = bbox_main.y0
                y_mid_lo = bbox_strip.y1
            else:
                y_mid_hi = bbox_main.y1
                y_mid_lo = bbox_strip.y0
            y_center = (y_mid_hi + y_mid_lo) / 2
            y_delta = abs(y_mid_hi - y_mid_lo) * 0.25
            fig.lines.extend([
                plt.Line2D([x0, x1], [y_center + y_delta, y_center + y_delta], **kwargs),
                plt.Line2D([x0, x1], [y_center - y_delta, y_center - y_delta], **kwargs),
            ])

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved paired violins to {output_path}")


def create_rounds_time_violins(df: pd.DataFrame, output_path: str):
    """Two-panel paired violin (Rounds + Time) sharing one figure. Same
    styling as `create_paired_violins` panels but no disagreement strip —
    every paired session is plotted on the main axes regardless of Status.
    """
    import seaborn as sns
    sns.set_theme(style='ticks', context='notebook', font='DejaVu Sans')

    flat_color = '#B0B0B0'
    box_color = '#2B2B2B'
    protocols = ['Traditional', 'LLM']
    positions = [1.0, 2.0]
    dot_jitter = 0.045
    rng = np.random.default_rng(42)

    panels = [
        ('Round', 'Rounds', True),
        ('Time', 'Time (seconds)', True),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plt.subplots_adjust(wspace=0.22)

    for ax, (metric, ylabel, lower_is_better) in zip(axes, panels):
        paired = build_paired_protocol_data(df, metric)
        if paired is None or len(paired) == 0:
            print(f"No paired data for {metric}; skipping panel")
            continue

        y_alt = paired['Traditional'].values.astype(float)
        y_part = paired['LLM'].values.astype(float)
        n = len(y_alt)

        test_name, p_value, effect_size = perform_pairwise_comparison(
            y_alt.tolist(), y_part.tolist()
        )
        if p_value < 0.001:
            stars = '***'
        elif p_value < 0.01:
            stars = '**'
        elif p_value < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'

        if n >= 2:
            parts = ax.violinplot([y_alt, y_part], positions=positions, widths=0.9,
                                  showmeans=False, showmedians=False, showextrema=False)
            for body, protocol, pos in zip(parts['bodies'], protocols, positions):
                verts = body.get_paths()[0].vertices
                if protocol == 'Traditional':
                    verts[:, 0] = np.clip(verts[:, 0], -np.inf, pos)
                else:
                    verts[:, 0] = np.clip(verts[:, 0], pos, np.inf)
                body.set_facecolor(PROTOCOL_COLORS[protocol])
                body.set_alpha(0.35)
                body.set_edgecolor('none')

        delta = y_part - y_alt
        posa_better = delta < 0 if lower_is_better else delta > 0
        complete_better = delta > 0 if lower_is_better else delta < 0
        line_colors = np.where(posa_better, PROTOCOL_COLORS['LLM'],
                       np.where(complete_better, PROTOCOL_COLORS['Traditional'], flat_color))

        x_alt = positions[0] + rng.uniform(-dot_jitter, dot_jitter, n)
        x_part = positions[1] + rng.uniform(-dot_jitter, dot_jitter, n)

        for i in range(n):
            ax.plot([x_alt[i], x_part[i]], [y_alt[i], y_part[i]],
                    color=line_colors[i], alpha=0.28, linewidth=0.7, zorder=2,
                    solid_capstyle='round')

        ax.scatter(x_alt, y_alt, c=line_colors, s=16, alpha=0.85,
                   edgecolor='white', linewidth=0.4, zorder=4)
        ax.scatter(x_part, y_part, c=line_colors, s=16, alpha=0.85,
                   edgecolor='white', linewidth=0.4, zorder=4)

        if n >= 2:
            bp = ax.boxplot([y_alt, y_part], positions=positions, widths=0.18,
                            patch_artist=True, showfliers=False, zorder=3,
                            medianprops=dict(color=box_color, linewidth=2.2),
                            whiskerprops=dict(color=box_color, linewidth=1.2),
                            capprops=dict(color=box_color, linewidth=1.2))
            for patch in bp['boxes']:
                patch.set_facecolor('white')
                patch.set_edgecolor(box_color)
                patch.set_linewidth(1.3)

            y_hi = max(y_alt.max(), y_part.max())
            y_lo = min(y_alt.min(), y_part.min())
            span = max(y_hi - y_lo, 1e-6)
            bracket_y = y_hi + span * 0.10
            tick_y = bracket_y - span * 0.02
            ax.plot([positions[0], positions[0], positions[1], positions[1]],
                    [tick_y, bracket_y, bracket_y, tick_y],
                    color=box_color, linewidth=1.2, zorder=7)
            ax.text(sum(positions) / 2, bracket_y + span * 0.015, stars,
                    ha='center', va='bottom', fontsize=14, fontweight='bold',
                    color=box_color)
            ax.set_ylim(y_lo - span * 0.05, bracket_y + span * 0.14)

        ax.text(0.98, 0.98, f'd = {abs(effect_size):.2f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=11, color='#555555')

        ax.set_xticks(positions)
        ax.set_xticklabels(protocols, fontsize=14, fontweight='bold')
        ax.set_xlim(0.35, 2.65)
        ax.set_ylabel(ylabel, fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', labelsize=13)

        sns.despine(ax=ax, trim=False)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved rounds+time violins to {output_path}")


def _apply_mixed_inversion(values_list: list) -> list:
    if len(values_list) == 0:
        return values_list
    if len(values_list) <= 2:
        return values_list[::-1]
    mid = len(values_list) // 2
    if len(values_list) % 2 == 0:
        return values_list[mid:] + values_list[:mid]
    left = values_list[:mid]
    right = values_list[mid + 1:]
    middle = [values_list[mid]]
    return right + middle + left


def _build_canonical_profiles(tmp_dir: str) -> Tuple[str, str]:
    """Mirror how profiles are actually built in production:
      - Issue weights are fixed per domain (same for every user).
      - Value utilities are rank-based: (N - rank + 1) / N.
      - LLM issue weights swap Accommodation <-> Activities.
      - LLM value utilities are inverted: full reverse for
        Destination/Season, Mixed rotation for the rest.
    The bid-space topology only depends on these rules (not on which
    specific value names each user ranks where), so a single canonical
    space is representative of every user."""
    import json

    template_path = os.path.join(
        PROJECT_ROOT, "main", "domains", "englisch", "holiday", "profile.json"
    )
    with open(template_path, "r") as f:
        template = json.load(f)

    human_weights = template["issueWeights"]
    human_issues = {}
    for issue_name, values_dict in template["issues"].items():
        names = list(values_dict.keys())
        n = len(names)
        human_issues[issue_name] = {
            name: (n - rank) / n
            for rank, name in enumerate(names)
        }

    human_profile = {
        "reservationValue": 0,
        "issueWeights": human_weights,
        "issues": human_issues,
    }

    llm_weights = {
        "Destination": 0.30,
        "Season": 0.25,
        "Activities": 0.20,
        "Accommodation": 0.15,
        "Transportation": 0.10,
    }
    fully_inverted_issues = {"Destination", "Season"}

    llm_issues = {}
    for issue_name, values_dict in human_issues.items():
        sorted_items = sorted(values_dict.items(), key=lambda x: x[1], reverse=True)
        names = [k for k, _ in sorted_items]
        utils = [v for _, v in sorted_items]
        inv = utils[::-1] if issue_name in fully_inverted_issues else _apply_mixed_inversion(utils)
        llm_issues[issue_name] = {names[i]: inv[i] for i in range(len(names))}

    llm_profile = {
        "reservationValue": 0,
        "issueWeights": llm_weights,
        "issues": llm_issues,
    }

    human_path = os.path.join(tmp_dir, "human_profile.json")
    llm_path = os.path.join(tmp_dir, "llm_profile.json")
    with open(human_path, "w") as f:
        json.dump(human_profile, f)
    with open(llm_path, "w") as f:
        json.dump(llm_profile, f)
    return human_path, llm_path


@functools.lru_cache(maxsize=1)
def _canonical_bid_space_data():
    """Build the canonical holiday bid space once and cache it.
    Returns (xs, ys, pareto_xs, pareto_ys, nash_h, nash_a) — all tuples
    so the lru_cache is happy."""
    import tempfile

    script_dir = os.path.dirname(os.path.abspath(__file__))
    removed = script_dir in sys.path
    if removed:
        sys.path.remove(script_dir)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    try:
        from main.nenv.Preference import Preference
        from main.nenv.BidSpace import BidSpace
    finally:
        if removed:
            sys.path.insert(0, script_dir)

    with tempfile.TemporaryDirectory() as tmp:
        human_path, llm_path = _build_canonical_profiles(tmp)
        pref_user = Preference(human_path)
        pref_llm = Preference(llm_path)
        bid_space = BidSpace(pref_user, pref_llm)

        points = bid_space.bid_points
        xs = tuple(bp.utility_a for bp in points)
        ys = tuple(bp.utility_b for bp in points)

        pareto = sorted(bid_space.pareto, key=lambda bp: bp.utility_a)
        px = tuple(bp.utility_a for bp in pareto)
        py = tuple(bp.utility_b for bp in pareto)

        nash = bid_space.nash_point
        return xs, ys, px, py, float(nash.utility_a), float(nash.utility_b)


def _canonical_nash_point() -> Tuple[float, float]:
    xs, ys, px, py, nash_h, nash_a = _canonical_bid_space_data()
    return nash_h, nash_a


def create_bidspace_only(output_path: str):
    """
    Bare bid-space plot: all bids (gray), Pareto frontier, and the Nash
    point. No title, no agreements, no distributions.
    """
    xs_t, ys_t, px_t, py_t, nash_h, nash_a = _canonical_bid_space_data()
    xs = list(xs_t)
    ys = list(ys_t)
    px = list(px_t)
    py = list(py_t)

    import seaborn as sns
    sns.set_theme(style='ticks', context='notebook', font='DejaVu Sans')

    bid_color = '#3B7FB6'
    pareto_color = '#2C3E50'
    nash_color = '#009E73'

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(xs, ys, s=10, alpha=0.55, color=bid_color,
               edgecolor='white', linewidths=0.2,
               label=f"All bids (n={len(xs)})", zorder=1)
    ax.plot(px, py, color=pareto_color, linewidth=1.4, alpha=0.9,
            label="Pareto frontier", zorder=2)

    x_lo, x_hi = min(xs + [nash_h]), max(xs + [nash_h])
    y_lo, y_hi = min(ys + [nash_a]), max(ys + [nash_a])
    x_pad = (x_hi - x_lo) * 0.04
    y_pad = (y_hi - y_lo) * 0.04
    ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    ax.plot([nash_h, nash_h], [ax.get_ylim()[0], nash_a],
            color=nash_color, linestyle='--', linewidth=2.4, alpha=0.85, zorder=3)
    ax.plot([ax.get_xlim()[0], nash_h], [nash_a, nash_a],
            color=nash_color, linestyle='--', linewidth=2.4, alpha=0.85, zorder=3)

    ax.plot(nash_h, nash_a, marker='^',
            markersize=20, color=nash_color, markeredgecolor='white',
            markeredgewidth=1.6, linestyle='none',
            label=f"Nash (H={nash_h:.2f} A={nash_a:.2f})",
            zorder=5)

    ax.set_xlabel("Human utility", fontsize=18, fontweight='bold')
    ax.set_ylabel("Agent utility", fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=15)
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')

    ax.legend(frameon=False, labelspacing=0.6, fontsize=14,
              loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved bid-space to {output_path}")


def create_bidspace_with_agreements(df: pd.DataFrame, output_path: str):
    """
    Single-panel bid-space plot with both protocols overlaid.
    Blue X = LLM agreements, orange X = Traditional agreements.
    Marginal KDEs along the x and y axes are drawn twice (one per
    protocol) and overlapped with transparency.
    """
    xs_t, ys_t, px_t, py_t, nash_h, nash_a = _canonical_bid_space_data()
    xs = list(xs_t)
    ys = list(ys_t)
    px = list(px_t)
    py = list(py_t)

    import seaborn as sns
    from scipy.stats import gaussian_kde

    sns.set_theme(style='ticks', context='notebook', font='DejaVu Sans')

    bid_color = '#C8CFD9'
    pareto_color = '#6B7280'
    nash_color = '#009E73'
    protocol_colors = PROTOCOL_COLORS
    protocol_display_names = {'LLM': 'Gemini', 'Traditional': 'Hybrid'}

    success_only = df[df['Status'] == 'Success']
    x_min = min(min(xs), success_only['HumanUtility'].min())
    x_max = max(max(xs), success_only['HumanUtility'].max())
    y_min = min(min(ys), success_only['AgentUtility'].min())
    y_max = max(max(ys), success_only['AgentUtility'].max())
    x_pad = (x_max - x_min) * 0.04
    y_pad = (y_max - y_min) * 0.04
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    ribbon_frac = 0.22

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.scatter(xs, ys, s=4, alpha=0.35, color=bid_color, zorder=1)
    ax.plot(px, py, color=pareto_color, linewidth=1.4, alpha=0.9, zorder=2)

    xg = np.linspace(xlim[0], xlim[1], 300)
    yg = np.linspace(ylim[0], ylim[1], 300)

    for protocol in ['Traditional', 'LLM']:
        success = df[(df['Protocol'] == protocol) & (df['Status'] == 'Success')]
        if len(success) == 0:
            continue
        color = protocol_colors[protocol]

        kde_h = gaussian_kde(success['HumanUtility'], bw_method=0.35)
        kde_a = gaussian_kde(success['AgentUtility'], bw_method=0.35)
        dh = kde_h(xg)
        da = kde_a(yg)
        dh_top = ylim[0] + (dh / dh.max()) * ribbon_frac * y_span
        da_right = xlim[0] + (da / da.max()) * ribbon_frac * x_span

        ax.fill_between(xg, ylim[0], dh_top, color=color,
                        alpha=0.18, linewidth=0, zorder=0)
        ax.plot(xg, dh_top, color=color, alpha=0.6,
                linewidth=1.4, zorder=0)
        ax.fill_betweenx(yg, xlim[0], da_right, color=color,
                         alpha=0.18, linewidth=0, zorder=0)
        ax.plot(da_right, yg, color=color, alpha=0.6,
                linewidth=1.4, zorder=0)

        ax.scatter(success['HumanUtility'], success['AgentUtility'],
                   marker='X', s=220, c=color,
                   edgecolor='white', linewidths=1.2,
                   alpha=0.78, zorder=5,
                   label=protocol_display_names[protocol])

    ax.plot([nash_h, nash_h], [ylim[0], nash_a],
            color=nash_color, linestyle='--', linewidth=2.4, alpha=0.85, zorder=3)
    ax.plot([xlim[0], nash_h], [nash_a, nash_a],
            color=nash_color, linestyle='--', linewidth=2.4, alpha=0.85, zorder=3)

    ax.plot(nash_h, nash_a, marker='^',
            markersize=20, color=nash_color, markeredgecolor='white',
            markeredgewidth=1.6, linestyle='none',
            label="Nash",
            zorder=6)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Human utility", fontsize=22, fontweight='bold')
    ax.set_ylabel("Agent utility", fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18)
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')

    ax.legend(frameon=False, labelspacing=0.7, fontsize=28,
              loc='upper right', markerscale=1.4, handletextpad=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved bid-space w/ agreements to {output_path}")


def create_utility_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a table of agent and human utilities across protocols with mean ± std.

    Args:
        df: DataFrame with session data

    Returns:
        DataFrame with utilities formatted as "mean ± std"
    """
    rows = []

    for protocol in PROTOCOL_ORDER:
        protocol_df = df[df['Protocol'] == protocol]

        human_mean, human_std = summarize_series(protocol_df['HumanUtility'])
        agent_mean, agent_std = summarize_series(protocol_df['AgentUtility'])
        dist_mean, dist_std = summarize_series(protocol_df['NashDistance'])

        rows.append({
            'Protocol': protocol,
            'Human Utility': f'{human_mean:.3f} ± {human_std:.3f}',
            'Agent Utility': f'{agent_mean:.3f} ± {agent_std:.3f}',
            'Nash Distance': f'{dist_mean:.3f} ± {dist_std:.3f}',
            'N': len(protocol_df)
        })

    return pd.DataFrame(rows)


def run_significance_tests(df: pd.DataFrame, domain_filter: str = None) -> pd.DataFrame:
    """
    Run significance tests comparing Traditional vs LLM protocols.

    Args:
        df: DataFrame with session data
        domain_filter: Optional domain to filter by (unused in the current analysis)

    Returns:
        DataFrame with test results for each metric
    """
    if domain_filter:
        df = df[df['Domain'] == domain_filter]

    results = []
    metrics = METRICS + SIDE_METRICS

    for metric in metrics:
        paired_df = build_paired_protocol_data(df, metric)
        alt_mean, alt_std = summarize_series(df[df['Protocol'] == 'Traditional'][metric])
        part_mean, part_std = summarize_series(df[df['Protocol'] == 'LLM'][metric])

        if np.isnan(alt_mean) or np.isnan(part_mean):
            results.append({
                'Metric': metric,
                'Traditional': f'{alt_mean:.3f} ± {alt_std:.3f}',
                'LLM': f'{part_mean:.3f} ± {part_std:.3f}',
                'Best': 'N/A',
                'Test': 'Insufficient data',
                'p-value': np.nan,
                'Significant': 'No',
                "Cohen's d": np.nan,
                'Effect': 'N/A',
                'Paired_N': 0
            })
            continue

        if metric in LOWER_IS_BETTER:
            best = 'Traditional' if alt_mean < part_mean else 'LLM'
        else:
            best = 'Traditional' if alt_mean > part_mean else 'LLM'

        if paired_df is None or len(paired_df) < 2:
            test_name = 'Insufficient paired data'
            p_value = np.nan
            effect_size = np.nan
            effect_interp = 'N/A'
            paired_n = 0 if paired_df is None else len(paired_df)
        else:
            test_name, p_value, effect_size = perform_pairwise_comparison(
                paired_df['Traditional'].tolist(),
                paired_df['LLM'].tolist()
            )
            effect_interp = interpret_effect_size(effect_size)
            paired_n = len(paired_df)

        results.append({
            'Metric': metric,
            'Traditional': f'{alt_mean:.3f} ± {alt_std:.3f}',
            'LLM': f'{part_mean:.3f} ± {part_std:.3f}',
            'Best': best,
            'Test': test_name,
            'p-value': round(p_value, 4) if pd.notna(p_value) else np.nan,
            'Significant': 'Yes' if pd.notna(p_value) and p_value < 0.05 else 'No',
            "Cohen's d": round(effect_size, 3) if pd.notna(effect_size) else np.nan,
            'Effect': effect_interp,
            'Paired_N': paired_n
        })

    return pd.DataFrame(results)


def save_excel_summary(raw_df: pd.DataFrame, overall_stats: pd.DataFrame,
                       success_rates: pd.DataFrame, output_path: str):
    """
    Save all analysis results to a multi-sheet Excel workbook.

    Args:
        raw_df: Raw session data
        overall_stats: Overall statistics DataFrame
        success_rates: Success rates DataFrame
        output_path: Path to save Excel file
    """
    with pd.ExcelWriter(output_path) as writer:
        # Sheet 1: Raw Data
        raw_df.to_excel(writer, sheet_name='Raw_Data', index=False)

        # Sheet 2: Overall Statistics
        overall_pivot = overall_stats.pivot_table(
            index='Metric',
            columns='Protocol',
            values=['Count', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Q1', 'Q3']
        )
        overall_pivot.to_excel(writer, sheet_name='Overall_Statistics')

        # Sheet 3: Success Rates
        success_rates.to_excel(writer, sheet_name='Success_Rates', index=False)

        # Sheet 4: Overall Utility Table (mean ± std)
        overall_utility_table = create_utility_table(raw_df)
        overall_utility_table.to_excel(writer, sheet_name='Utilities_Overall', index=False)

        # Sheet 5: Overall Significance Tests
        overall_sig = run_significance_tests(raw_df)
        overall_sig.to_excel(writer, sheet_name='SigTests_Overall', index=False)

    print(f"Saved Excel summary to {output_path}")


def run_analysis(df: pd.DataFrame, label: str, out_dir: str):
    figures_dir = os.path.join(out_dir, "figures")
    csv_dir = os.path.join(out_dir, "csv_exports")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[{label}] Computing statistics and tests...")
    print(f"{'='*60}")

    overall_stats = compute_descriptive_stats(df, ['Protocol'], METRICS + SIDE_METRICS)
    success_rates = compute_success_rates(df)
    overall_sig = run_significance_tests(df)

    print(f"[{label}] Significance (Traditional vs LLM):")
    for _, row in overall_sig.iterrows():
        sig_marker = "*" if row['Significant'] == 'Yes' else ""
        cohens_d = row["Cohen's d"]
        print(
            f"  {row['Metric']}: Best={row['Best']}, p={row['p-value']}{sig_marker} "
            f"({row['Test']}, paired_n={row['Paired_N']}, d={cohens_d} {row['Effect']})"
        )

    create_paired_violins(df, os.path.join(figures_dir, "paired_violins.png"))
    create_rounds_time_violins(
        df, os.path.join(figures_dir, "paired_violins_rounds_time.png")
    )
    create_bidspace_only(os.path.join(figures_dir, "bid_space.png"))
    create_bidspace_with_agreements(df, os.path.join(figures_dir, "bidspace_agreements.png"))

    save_excel_summary(
        df,
        overall_stats,
        success_rates,
        os.path.join(out_dir, "protocol_comparison_results.xlsx")
    )

    df.to_csv(os.path.join(csv_dir, "raw_agreements.csv"), index=False)
    overall_stats['Comparison'] = 'Overall'
    overall_stats.to_csv(os.path.join(csv_dir, "summary_statistics.csv"), index=False)
    success_rates.to_csv(os.path.join(csv_dir, "success_rates.csv"), index=False)
    overall_sig.to_csv(os.path.join(csv_dir, "significance_tests.csv"), index=False)
    print(f"[{label}] Saved results to {out_dir}")


def main():
    print("="*60)
    print("Protocol Comparison Analysis")
    print("="*60)

    raw_data = load_session_files(SESSION_LOGS_DIR)

    failed = raw_data[raw_data['Status'] == 'Failed']
    print(f"\n{'='*60}")
    print(f"Failed sessions ({len(failed)}):")
    print(f"{'='*60}")
    for _, row in failed.sort_values(['UserID', 'Protocol']).iterrows():
        print(f"  {row['UserID']:<30s} {row['Protocol']:<12s} rounds={int(row['Round']):<3d} {row['Filename']}")

    all_paired = pair_on_protocol(raw_data, "ALL")
    accepted_only = pair_on_protocol(
        raw_data[raw_data['Status'] == 'Success'].copy(),
        "ACCEPTED"
    )

    run_analysis(accepted_only, "ACCEPTED", os.path.join(OUTPUT_DIR, "accepted_only"))
    run_analysis(all_paired, "ALL", os.path.join(OUTPUT_DIR, "all_sessions"))

    users_df = pd.DataFrame({
        'UserID': sorted(all_paired['UserID'].unique(), key=str.casefold)
    })
    users_csv_path = os.path.join(OUTPUT_DIR, "users.csv")
    users_df.to_csv(users_csv_path, index=False)
    print(f"\nWrote {len(users_df)} paired users to {users_csv_path}")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
