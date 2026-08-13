import argparse
import math
from pathlib import Path

import pandas as pd
from matplotlib import cm, colors
import matplotlib.pyplot as plt


def run(log_dir, standardized=False, output_path=None, poll_interval=1.0):
    '''Monitor a result log and keep the sample plots updated until interrupted.'''
    log_dir = Path(log_dir)
    result_path = log_dir / 'result.csv'
    if not result_path.exists():
        raise FileNotFoundError(f'Could not find result.csv in {log_dir}')
    plt.ion()

    print(f'Monitoring {result_path}. Press Ctrl+C to terminate.')
    previous_state = None
    figures = []
    try:
        while True:
            current_state = _file_state(result_path)
            if current_state != previous_state:
                previous_state = current_state
                df = pd.read_csv(result_path)
                figures = plot_samples(df, standardized, figures=figures)
                if output_path is not None:
                    save_figures(figures, output_path)

            if figures and not any(plt.fignum_exists(fig.number) for fig, _ in figures):
                break
            plt.pause(poll_interval)
    except KeyboardInterrupt:
        print('Terminated')
    finally:
        plt.ioff()


def _file_state(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size

def _close_figures(figures):
    for fig, _ in figures:
        plt.close(fig)


def _sanitize_name(name):
    return ''.join(c if c.isalnum() or c in ('-', '_') else '-' for c in name)


def save_figures(figures, output_path):
    '''Save figures to one or more output files.'''
    output_path = Path(output_path)
    if len(figures) == 1:
        figures[0][0].savefig(output_path, dpi=150)
        return
    for fig, suffix in figures:
        fig.savefig(
            output_path.with_name(
                f'{output_path.stem}-{_sanitize_name(suffix)}{output_path.suffix}'
            ),
            dpi=150,
        )


def plot_samples(df, standardized, figures=None):
    '''Create or update sample plots from a result.csv DataFrame.'''
    # pylint: disable=too-many-statements, too-many-locals
    tag = "standardized" if standardized else "raw"
    df = df[df["tag"] == tag]
    metrics = [s for s in df.columns if s.startswith("metric_")]
    params = [s for s in df.columns if s.startswith("param_")]
    step = df["step"].values

    if len(df) == 0 or len(metrics) == 0 or len(params) == 0:
        return []

    nplots = len(params)
    nrows = math.floor(math.sqrt(nplots))
    ncols = math.ceil(nplots / nrows)
    figsize = (2 + 5 * ncols, 5 * nrows)

    cmap = plt.colormaps['viridis']
    norm = colors.Normalize(vmin=min(step), vmax=max(min(step) + 1, step))

    existing_figures = {} if figures is None else {suffix : fig for fig, suffix in figures}
    figures = []
    for metric in metrics:
        suffix = metric[7:]
        fig = existing_figures.pop(suffix, None)
        state = getattr(fig, '_daisypy_state', None) if fig is not None else None
        reusable = (
            fig is not None and
            state is not None and
            state['params'] == tuple(params) and
            state['shape'] == (nrows, ncols)
        )

        if not reusable:
            if fig is not None:
                plt.close(fig)
            fig = plt.figure(figsize=figsize, layout="constrained")
            axs = fig.subplots(
                nrows,
                ncols,
                squeeze=False,
                sharex=standardized,
                sharey=True,
            )
            scatter_by_param = {}
            row = 0
            col = 0
            for param in params:
                if col == ncols:
                    row += 1
                    col = 0
                axis = axs[row][col]
                plot = axis.scatter(
                    df[param],
                    df[metric],
                    c=step,
                    cmap=cmap,
                    norm=norm,
                    marker='+',
                )
                if col == 0:
                    axis.set_ylabel(metric[7:])
                axis.set_xlabel(param[6:])
                scatter_by_param[param] = plot
                col += 1
            for extra_axis in axs.flat[len(params):]:
                extra_axis.set_visible(False)
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            colorbar = fig.colorbar(
                mappable,
                ax=fig.get_axes(),
                label="run",
            )
            state = {
                'params' : tuple(params),
                'shape' : (nrows, ncols),
                'scatter_by_param' : scatter_by_param,
                'mappable' : mappable,
                'colorbar' : colorbar,
            }
            fig._daisypy_state = state  # pylint: disable=protected-access
        else:
            fig.set_size_inches(figsize, forward=True)
            state['mappable'].set_norm(norm)
            state['mappable'].set_cmap(cmap)
            state['colorbar'].update_normal(state['mappable'])
            for param in params:
                scatter = state['scatter_by_param'][param]
                scatter.set_offsets(df[[param, metric]].to_numpy())
                scatter.set_array(step)
                scatter.set_norm(norm)
                scatter.set_cmap(cmap)

        fig.suptitle(f"Sampled parameters vs {suffix}")
        fig.canvas.draw_idle()
        figures.append((fig, suffix))
    for fig in existing_figures.values():
        plt.close(fig)
    plt.show(block=False)
    return figures


def main():
    '''Entry point.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', type=str, help='Path to optimization log directory')
    parser.add_argument(
        '--standardized', action="store_true", default=False,
        help="If set plot standardized parameter values otherwise plot actual parameter values"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional output image path. If multiple metrics exist, the metric name is appended.',
    )
    parser.add_argument(
        '--poll-interval',
        type=float,
        default=15.0,
        help='Seconds between checks for updates to result.csv.',
    )
    args = parser.parse_args()
    run(args.log_dir, args.standardized, args.output, args.poll_interval)


if __name__ == '__main__':
    main()
