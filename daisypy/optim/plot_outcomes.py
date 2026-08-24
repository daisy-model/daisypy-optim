import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import animation
from matplotlib import cm
from matplotlib import colors


def sanitize_name(name):
    '''Return a filesystem-friendly version of a name.'''
    return ''.join(c if c.isalnum() or c in ('-', '_') else '-' for c in name)

def _ensure_step_index_columns(data):
    '''Return a copy with explicit ``step`` and ``index`` columns.'''
    data = data.copy()
    if 'step' in data.columns and 'index' in data.columns:
        return data
    raise ValueError("Expected 'step' and 'index' columns")


def _prepare_objective_data(data, outcome_name, max_curves=None, targets=None):
    '''Prepare grouped outcome data and run-based coloring metadata.'''
    # pylint: disable=too-many-locals
    objective_data = _ensure_step_index_columns(data)
    objective_data = objective_data[objective_data['outcome_name'] == outcome_name].copy()
    objective_data['time'] = pd.to_datetime(objective_data['time'])
    grouped = list(objective_data.groupby(['step', 'index'], sort=False))
    if max_curves is not None:
        grouped = grouped[:max_curves]
    target_data = None
    if targets is not None:
        target_data = targets[targets['outcome_name'] == outcome_name].copy()
        if len(target_data) == 0:
            target_data = None
        else:
            target_data['time'] = pd.to_datetime(target_data['time'])

    evaluation_groups = sorted({int(step) for (step, _), _ in grouped})
    min_group = min(evaluation_groups)
    max_group = max(evaluation_groups)
    if min_group == max_group:
        norm = colors.Normalize(vmin=min_group, vmax=min_group + 1)
    else:
        norm = colors.Normalize(vmin=min_group, vmax=max_group)
    cmap = plt.colormaps['viridis']
    color_lookup = {
        evaluation_group: cmap(norm(evaluation_group))
        for evaluation_group in evaluation_groups
    }
    x_values = [objective_data['time']]
    y_values = [objective_data['predicted_value']]
    if target_data is not None:
        x_values.append(target_data['time'])
        y_values.append(target_data['target_value'])
    xlim = (min(series.min() for series in x_values), max(series.max() for series in x_values))
    ylim = (min(series.min() for series in y_values), max(series.max() for series in y_values))
    run_groups = {}
    for key, group in grouped:
        run = int(key[0])
        run_groups.setdefault(run, []).append((key, group))
    return grouped, run_groups, color_lookup, cmap, norm, xlim, ylim, target_data


def _style_axes(ax, title, xlim, ylim):
    '''Apply common axis styling.'''
    ax.set_title(title)
    ax.set_xlabel('time')
    ax.set_ylabel('predicted value')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def _draw_target(ax, target_data):
    '''Draw target values if available.'''
    if target_data is not None:
        ax.plot(
            target_data['time'],
            target_data['target_value'],
            color='red',
            marker='o',
            linewidth=0,
        )


def plot_objective_curves(data, outcome_name, output_path=None, max_curves=None, targets=None):
    '''Plot all outcome curves for a single objective.'''
    # pylint: disable=too-many-locals
    grouped, _, color_lookup, cmap, norm, xlim, ylim, target_data = _prepare_objective_data(
        data, outcome_name, max_curves=max_curves, targets=targets
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    for (step, _), group in grouped:
        color = color_lookup[int(step)]
        ax.plot(
            group['time'],
            group['predicted_value'],
            color=color,
            alpha=0.35,
            linewidth=1.0,
        )

    _draw_target(ax, target_data)
    _style_axes(ax, f'{outcome_name} outcome curves ({len(grouped)} evaluations)', xlim, ylim)
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        label=f'run ({int(norm.vmin)} to {int(norm.vmax)})'
    )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    return fig, ax


def animate_objective_curves(
        data, outcome_name, output_path=None, max_curves=None, fps=2, targets=None
):
    '''Animate one run per frame, keeping the previous run visible with reduced alpha.'''
    # pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments
    _, run_groups, color_lookup, cmap, norm, xlim, ylim, target_data = _prepare_objective_data(
        data, outcome_name, max_curves=max_curves, targets=targets
    )
    runs = sorted(run_groups)

    fig, ax = plt.subplots(figsize=(10, 5))
    _style_axes(ax, f'{outcome_name} run {runs[0]}', xlim, ylim)
    _draw_target(ax, target_data)
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        label=f'run ({int(norm.vmin)} to {int(norm.vmax)})'
    )

    def draw_run(run, alpha, linestyle='solid', color=None):
        for (step, _), group in run_groups[run]:
            if color is None:
                color = color_lookup[int(step)]
            ax.plot(
                group['time'],
                group['predicted_value'],
                color=color,
                alpha=alpha,
                linewidth=1,
                linestyle=linestyle
            )

    def update(frame):
        ax.clear()
        previous_runs = runs[:frame] if frame > 0 else []
        for i, previous_run in enumerate(previous_runs, start=1):
            alpha = i / len(previous_runs) * 0.5
            draw_run(previous_run, alpha=alpha, linestyle='dashed', color='gray')
        draw_run(runs[frame], alpha=1.0)
        _draw_target(ax, target_data)
        _style_axes(ax, f'{outcome_name} run {runs[frame]}', xlim, ylim)

    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=len(runs),
        interval=max(200, 2000 // max(len(runs), 1)),
        repeat=False,
    )
    if output_path is not None:
        ani.save(output_path, writer=animation.PillowWriter(fps=fps))
        plt.close(fig)
    return ani


def main():
    '''Entry point.'''
    parser = argparse.ArgumentParser(description='Plot curves from an outcomes.csv log')
    parser.add_argument('outcomes_csv', type=Path, help='Path to outcomes.csv')
    parser.add_argument(
        '--objective',
        action='append',
        dest='objectives',
        help='Objective name to plot. Can be given multiple times.',
    )
    parser.add_argument(
        '--max-curves',
        type=int,
        default=None,
        help='Limit the number of evaluation curves plotted per objective.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Optional output image path. If multiple objectives are plotted, the objective name is'
             ' appended to the file name.',
    )
    parser.add_argument(
        '--animate',
        action='store_true',
        help='Create an animation with one frame per step.',
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=2,
        help='Frames per second when saving an animation.',
    )
    parser.add_argument(
        '--targets-csv',
        type=Path,
        default=None,
        help='Optional path to targets.csv. Defaults to a targets.csv next to outcomes.csv.',
    )
    args = parser.parse_args()

    data = pd.read_csv(args.outcomes_csv)
    targets_path = args.targets_csv
    if targets_path is None:
        candidate = args.outcomes_csv.with_name('targets.csv')
        if candidate.exists():
            targets_path = candidate
    targets = pd.read_csv(targets_path) if (
        targets_path is not None and targets_path.exists()
    ) else None
    outcome_names = args.objectives
    if outcome_names is None:
        outcome_names = list(data['outcome_name'].drop_duplicates())

    animations = []
    for outcome_name in outcome_names:
        output_path = None
        if args.output is not None:
            if len(outcome_names) == 1:
                output_path = args.output
            else:
                output_path = args.output.with_name(
                    f'{args.output.stem}-{sanitize_name(outcome_name)}{args.output.suffix}'
                )
        if args.animate:
            ani = animate_objective_curves(
                data,
                outcome_name=outcome_name,
                output_path=output_path,
                max_curves=args.max_curves,
                fps=args.fps,
                targets=targets,
            )
            if args.output is None:
                animations.append(ani)
        else:
            plot_objective_curves(
                data,
                outcome_name=outcome_name,
                output_path=output_path,
                max_curves=args.max_curves,
                targets=targets,
            )

    if args.output is None:
        plt.show()


if __name__ == '__main__':
    main()
