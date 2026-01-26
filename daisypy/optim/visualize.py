import warnings
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import Normalize
import matplotlib as mpl

__all__ = [
    'plot_convergence',
    'plot_result',
    'plot_result_1d',
    'plot_result_2d',
    'plot_result_3d',
    'plot_result_nd',
    'animate_result',
    'animate_result_1d',
    'animate_result_2d',
    'animate_result_nd',
]

def plot_convergence(df, step_var='step', f_var='objective_value', log_transform=True):
    '''Make a convergence plot of objective values.

    Parameters
    ----------
    df : pandas.DataFrame OR str
      If str assume it is a path to a csv file with optimization results

    step_var : str
      Name of step variable

    f_var : str
      Name of objective variable

    log_transform : bool
      If True apply a log transform to the objective value. The objective is shifted to be positive
      before log transforming.

    Returns
    -------
    (matplotlib.Figure, matplotlib.Axes)
    '''
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    step = df[step_var]
    f = df[f_var]
    if log_transform:
        if np.min(f) <= 0:
            f = 1e-8 + f - np.min(f)
        f = np.log(f)
    fig, ax = plt.subplots()
    even = step % 2 == 0
    odd = ~even
    ax.scatter(step[even], f[even], marker='.', c='black')
    ax.scatter(step[odd], f[odd], marker='+', c='black')
    ax.set_xlabel(step_var)
    ax.set_ylabel('log(f)')
    return fig, ax

def plot_result(df, step_var='step', f_var='objective_value'):
    '''Make a scatter plot of parameters and objective values at each optimization step. If there is
    more than 3 parameters, each parameter is plotted in its own subplot. If there is more than 20
    parameters, each set of 20 parameters is plotted in a separate figure.

    Parameters
    ----------
    df : pandas.DataFrame OR str
      If str assume it is a path to a csv file with optimization results

    step_var : str
      Name of step variable

    f_var : str
      Name of objective variable

    Returns
    -------
    (matplotlib.Figure, matplotlib.Axes) or list of (matplotlib.Figure, matplotlib.Axes)

    See also
    --------
    plot_result_1d, plot_result_2d, plot_result_3d, plot_result_nd
    '''
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    params = df.columns[~df.columns.isin({step_var, f_var})]
    match len(params):
        case 1: return plot_result_1d(df, params[0], step_var, f_var)
        case 2: return plot_result_2d(df, params[0], params[1], step_var, f_var)
        case 3: return plot_result_3d(df, params[0], params[1], params[2], f_var)
        case _: return plot_result_nd(df, params, step_var, f_var)

def plot_result_1d(df, var, step_var='step', f_var='objective_value'):
    '''Make a scatter plot of a single parameter and objective value at each optimization step.
    Parameters
    ----------
    df : pandas.DataFrame OR str
      If str assume it is a path to a csv file with optimization results

    var : str
      Name of parameter variable

    step_var : str
      Name of step variable

    f_var : str
      Name of objective variable

    Returns
    -------
    (matplotlib.Figure, matplotlib.Axes)

    See also
    --------
    plot_result, plot_result_2d, plot_result_3d, plot_result_nd
    '''
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    fig, ax = plt.subplots()
    x = df[step_var]
    y = df[var]
    f = df[f_var]
    transform, transform_text = _find_cmap_transform(f)
    c = transform(f)
    cm = ax.scatter(x, y, c=c)
    ax.set_xlabel(step_var)
    ax.set_ylabel(var)
    ax.set_title('Parameters at each step')
    fig.colorbar(cm, label=transform_text)
    fig.tight_layout()
    return fig, ax

def plot_result_2d(df, x_var, y_var, step_var='step', f_var='objective_value'):
    '''Make a scatter plot of two parameters and objective value at each optimization step.

    Parameters
    ----------
    df : pandas.DataFrame OR str
      If str assume it is a path to a csv file with optimization results

    x_var : str
      Name of parameter variable on x-axis

    y_var : str
      Name of parameter variable on y-axis

    step_var : str
      Name of step variable

    f_var : str
      Name of objective variable

    Returns
    -------
    (matplotlib.Figure, matplotlib.Axes)

    See also
    --------
    plot_result, plot_result_1d, plot_result_3d, plot_result_nd
    '''
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    fig, ax = plt.subplots()
    transform, transform_text = _find_cmap_transform(df[f_var])
    c = transform(df[f_var])
    norm = Normalize(vmin=np.min(c), vmax=np.max(c))
    grouped = pd.DataFrame({
        'x': df[x_var], 'y': df[y_var], 'c': c, 'step': df[step_var]
    }).groupby('step')
    for step, group in grouped:
        cm = ax.scatter(group['x'], group['y'], c=group['c'], marker=f'${step}$', norm=norm)
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_title(f'Parameters at each step for {len(grouped)} steps')
    fig.colorbar(cm, label=transform_text)
    fig.tight_layout()
    return fig, ax

def plot_result_3d(df, x_var, y_var, z_var, f_var):
    '''Make a 3D scatter plot of three parameters and objective values.

    Parameters
    ----------
    df : pandas.DataFrame OR str
      If str assume it is a path to a csv file with optimization results

    x_var : str
      Name of parameter variable on x-axis

    y_var : str
      Name of parameter variable on y-axis

    z_var : str
      Name of parameter variable on z-axis

    f_var : str
      Name of objective variable

    Returns
    -------
    (matplotlib.Figure, matplotlib.Axes)

    See also
    --------
    plot_result, plot_result_1d, plot_result_2d, plot_result_nd
    '''
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = df[x_var]
    y = df[y_var]
    z = df[z_var]
    f = df[f_var]
    transform, _ = _find_cmap_transform(f)
    c = transform(f)
    ax.scatter(x, y, z, c=c)
    return fig, ax

def plot_result_nd(df, params, step_var='step', f_var='objective_value', max_plots_in_figure=20):
    '''Plot each parameter in a separate subplot, showing the parameter value and objective value at
    each step. If there is more than 20 parameters, each set of 20 parameters is plotted in a
    separate figure.

    Parameters
    ----------
    df : pandas.DataFrame OR str
      If str assume it is a path to a csv file with optimization results

    params : list of str
      Names of parameter variables to plot

    step_var : str
      Name of step variable

    f_var : str
      Name of objective variable

    max_plots_in_figure : int
      Maximum number of parameters to plot per figure

    Returns
    -------
    (matplotlib.Figure, matplotlib.Axes) or list of (matplotlib.Figure, matplotlib.Axes)

    See also
    --------
    plot_result, plot_result_1d, plot_result_2d, plot_result_3d
    '''
    # pylint: disable=too-many-locals
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    n = len(params)
    if n > max_plots_in_figure:
        return [
            plot_result_nd(df, params[i:min(n, i+max_plots_in_figure)], step_var, f_var)
            for i in range(n, step=max_plots_in_figure)
        ]

    x = df[step_var]
    c, transform_text = _cmap_transform(df[f_var])

    rows = int(np.sqrt(n))
    cols = n // rows
    if rows * cols < n:
        cols += 1
    fig, axs = plt.subplots(rows, cols, squeeze=False)
    row, col = 0, 0
    for var in params:
        ax = axs[row,col]
        cm = ax.scatter(x, df[var], c=c)
        ax.set_xlabel(step_var)
        ax.set_ylabel(var)
        col = (col + 1) % cols
        if col == 0:
            row = row + 1
    fig.colorbar(cm, ax=axs, label=transform_text)
    return fig, ax

def animate_result(df, step_var='step', f_var='objective_value'):
    '''Make an animation showing parameters at each optimization step. If there are more than two
    parameters, two are selected at random and animated.

    Parameters
    ----------
    df : pandas.DataFrame OR str
      If str assume it is a path to a csv file with optimization results

    step_var : str
      Name of step variable

    f_var : str
      Name of objective variable

    Returns
    -------
    matplotlib.animation.AnimationFigure

    See also
    --------
    animate_result_1d, animate_result_2d, animate_result_nd
    '''
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    params = df.columns[~df.columns.isin({step_var, f_var})]

    match len(params):
        case 1: return animate_result_1d(df, params[0], step_var, f_var)
        case 2: return animate_result_2d(df, params[0], params[1], step_var, f_var)
        case _: return animate_result_nd(df, params, step_var, f_var)

def animate_result_1d(df, var, step_var='step', f_var='objective_value'):
    '''Make an animation showing a single parameter at each optimization step.

    Parameters
    ----------
    df : pandas.DataFrame OR str
      If str assume it is a path to a csv file with optimization results

    var : str
      Name of parameter variable

    step_var : str
      Name of step variable

    f_var : str
      Name of objective variable

    Returns
    -------
    matplotlib.animation.AnimationFigure

    See also
    --------
    animate_result, animate_result_2d, animate_result_nd
    '''
    # pylint: disable=too-many-locals
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    # TODO: Add boundary lines showing parameter extents
    cmap = mpl.colormaps[mpl.rcParams['image.cmap']]

    # Find x and y limits that cover full data set
    xlim = _compute_limits(df[step_var])
    ylim = _compute_limits(df[var])

    # Find a suitable transformation of the objective for coloring
    transform, transform_text = _find_cmap_transform(df[f_var])
    c = transform(df[f_var])
    norm = Normalize(vmin=np.min(c), vmax=np.max(c))

    # Prepare the data so we can just index it
    xs, ys, cs = [], [], []
    for step, group in df.groupby(step_var):
        xs.append([step]*len(group))
        ys.append(group[var])
        cs.append(cmap(norm(transform(group[f_var]))))

    fig, ax = plt.subplots(1)
    ax.set_xlabel(step_var)
    ax.set_ylabel(var)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Make a colormap that has correct labeling and values
    dummy_plot = ax.scatter([], [], c=[], cmap=cmap, norm=norm)
    fig.colorbar(dummy_plot, label=transform_text)

    plot = ax.scatter(xs[0], ys[0], c=cs[0])
    def update(frame):
        data = np.stack([xs[frame], ys[frame]]).T
        plot.set_offsets(data)
        plot.set_color(cs[frame])
    n_frames = len(xs)
    interval = max(100, 2000//n_frames)
    return animation.FuncAnimation(fig=fig, func=update, frames=n_frames, interval=interval)


def animate_result_2d(df, x_var, y_var, step_var='step', f_var='objective_value'):
    '''Make an animation showing two parameters at each optimization step.

    Parameters
    ----------
    df : pandas.DataFrame OR str
      If str assume it is a path to a csv file with optimization results

    x_var : str
      Name of parameter variable on x-axis

    y_var : str
      Name of parameter variable on y-axis

    step_var : str
      Name of step variable

    f_var : str
      Name of objective variable

    Returns
    -------
    matplotlib.animation.AnimationFigure

    See also
    --------
    animate_result, animate_result_1d, animate_result_nd
    '''
    # pylint: disable=too-many-locals
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    cmap = mpl.colormaps[mpl.rcParams['image.cmap']]

    # Find x and y limits that cover full data set
    xlim = _compute_limits(df[x_var])
    ylim = _compute_limits(df[y_var])

    # Find a suitable transformation of the objective for coloring
    transform, transform_text = _find_cmap_transform(df[f_var])
    c = transform(df[f_var])
    norm = Normalize(vmin=np.min(c), vmax=np.max(c))

    # Prepare the data so we can just index it
    steps, xs, ys, cs = [], [], [], []
    for step, group in df.groupby(step_var):
        steps.append(step)
        xs.append(group[x_var])
        ys.append(group[y_var])
        cs.append(cmap(norm(transform(group[f_var]))))

    fig, ax = plt.subplots(1)
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f'Step {steps[0]}')

    # Make a colormap that has correct labeling and values
    dummy_plot = ax.scatter([], [], c=[], cmap=cmap, norm=norm)
    fig.colorbar(dummy_plot, label=transform_text)

    plot = ax.scatter(xs[0], ys[0], c=cs[0])
    def update(frame):
        data = np.stack([xs[frame], ys[frame]]).T
        plot.set_offsets(data)
        plot.set_color(cs[frame])
        ax.set_title(f'Step {steps[frame]}')
    n_frames = len(xs)
    interval = max(100, 2000//n_frames)
    return animation.FuncAnimation(fig=fig, func=update, frames=n_frames, interval=interval)

def animate_result_nd(df, params, step_var='step', f_var='objective_value'):
    '''Make an animation of two parameters selected at random from a list of possible parameters

    Parameters
    ----------
    df : pandas.DataFrame OR str
      If str assume it is a path to a csv file with optimization results

    params : list of str
      Names of parameter variables

    step_var : str
      Name of step variable

    f_var : str
      Name of objective variable

    Returns
    -------
    matplotlib.animation.AnimationFigure

    See also
    --------
    animate_result, animate_result_1d, animate_result_nd
    '''

    warnings.warn("Can only animate 2 parameters at a time. Selecting two at random")
    params_list = list(params)
    random.shuffle(params_list)
    x_var = params_list[0]
    y_var = params_list[1]
    return animate_result_2d(df, x_var, y_var, step_var, f_var)

## Utilities
def _compute_limits(x):
    lim = [np.min(x), np.max(x)]
    w = (lim[1] - lim[0]) * 1.1
    c = (lim[0] + lim[1]) / 2
    return (c - w/2, c + w/2)

def _shift_and_scale(x):
    shifted = x - np.min(x)
    return shifted / np.max(shifted)

def _linearity(f):
    f = _shift_and_scale(f) # Standardize to [0,1]
    x = np.arange(len(f))
    n = x[-1]
    target = x / n * (f[n] - f[0]) + f[0] # Straight line connecting end points
    return np.mean((f - target))


def _cmap_transform(f):
    transform, transform_text = _find_cmap_transform(f)
    return transform(f), transform_text

def _find_cmap_transform(f):
    # Try to find a transformation of f such that it looks nice when used with linear color maps
    lin_score = _linearity(f)
    if lin_score < -0.2:
        return lambda x : np.log(1e-8 + x - np.min(f)), "log(f)"
    if lin_score < -0.1:
        return lambda x : np.sqrt(x - np.min(f)), "$\\sqrt{f}$"
    if lin_score > 0.2:
        return np.exp, "exp(f)"
    if lin_score > 0.1:
        return lambda x : x**2, "$f^2$"
    return lambda x : x, "f"
