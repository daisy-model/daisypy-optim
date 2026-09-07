# pylint: disable=too-many-lines
'''Local web app for monitoring optimization logs.'''
import logging
import threading
import webbrowser
from pathlib import Path

from dash import Dash, Input, Output, State, dcc, html, no_update
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from waitress import serve

_PAGE_STYLE = {
    'padding' : '12px 16px',
    'backgroundColor' : '#f5f6f7',
    'color' : '#2f3437',
    'fontFamily' : (
        '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif'
    ),
}
_PATH_STYLE = {
    'fontSize' : '12px',
    'color' : '#6c737a',
    'whiteSpace' : 'nowrap',
    'overflow' : 'hidden',
    'textOverflow' : 'ellipsis',
}
_STATUS_STYLE = {
    'marginTop' : '8px',
    'fontSize' : '12px',
    'color' : '#6c737a',
    'display' : 'flex',
    'gap' : '10px',
    'alignItems' : 'center',
    'flexWrap' : 'wrap',
    'paddingTop' : '6px',
    'borderTop' : '1px solid #d8dde3',
}
_TABS_STYLE = {
    'height' : '36px',
    'display' : 'flex',
    'alignItems' : 'end',
    'gap' : '4px',
    'borderBottom' : '1px solid #d8dde3',
}
_TAB_STYLE = {
    'padding' : '6px 12px',
    'height' : '34px',
    'lineHeight' : '20px',
    'fontSize' : '13px',
    'fontWeight' : '500',
    'backgroundColor' : '#eef0f2',
    'border' : '1px solid #d8dde3',
    'borderBottom' : 'none',
    'color' : '#4d545b',
    'borderTopLeftRadius' : '8px',
    'borderTopRightRadius' : '8px',
    'width' : 'auto',
    'display' : 'inline-flex',
    'alignItems' : 'center',
    'justifyContent' : 'center',
    'flex' : '0 0 auto',
}
_TAB_SELECTED_STYLE = {
    **_TAB_STYLE,
    'backgroundColor' : '#ffffff',
    'color' : '#22272b',
    'borderTop' : '2px solid #8c959f',
    'marginBottom' : '-1px',
}
_SECTION_STYLE = {
    'paddingTop' : '10px',
}
_CONTROLS_STYLE = {
    'display' : 'flex',
    'gap' : '12px',
    'alignItems' : 'end',
    'flexWrap' : 'wrap',
    'marginBottom' : '10px',
}
_FIELD_STYLE = {
    'display' : 'flex',
    'flexDirection' : 'column',
    'gap' : '4px',
}
_LABEL_STYLE = {
    'fontSize' : '12px',
    'fontWeight' : '500',
    'color' : '#4d545b',
}
_BUTTON_STYLE = {
    'height' : '32px',
    'padding' : '0 10px',
    'border' : '1px solid #d0d7de',
    'backgroundColor' : '#f8f9fa',
    'color' : '#57606a',
    'borderRadius' : '6px',
    'fontSize' : '12px',
}
_ICON_BUTTON_STYLE = {
    **_BUTTON_STYLE,
    'width' : '32px',
    'padding' : '0',
    'fontSize' : '13px',
}
_STATUS_ITEM_STYLE = {
    'display' : 'inline-flex',
    'alignItems' : 'center',
    'gap' : '5px',
}
_STATUS_DOT_STYLE = {
    'display' : 'inline-block',
    'width' : '7px',
    'height' : '7px',
    'borderRadius' : '50%',
    'backgroundColor' : '#6a9c78',
}
_STATUS_DOT_MISSING_STYLE = {
    **_STATUS_DOT_STYLE,
    'backgroundColor' : '#b26a6a',
}
_GRAPH_STYLE = {
    'height' : '82vh',
    'backgroundColor' : '#ffffff',
    'border' : '1px solid #d8dde3',
    'borderRadius' : '8px',
}
_UTILITY_BAR_STYLE = {
    'display' : 'flex',
    'justifyContent' : 'flex-start',
    'alignItems' : 'end',
    'gap' : '12px',
    'flexWrap' : 'wrap',
    'marginBottom' : '6px',
}
_REFRESH_CONTROL_STYLE = {
    'display' : 'flex',
    'alignItems' : 'flex-end',
    'gap' : '8px',
    'flexWrap' : 'nowrap',
}
_CHECKBOX_ROW_STYLE = {
    'display' : 'flex',
    'alignItems' : 'center',
    'gap' : '8px',
    'height' : '32px',
}
_CHECKBOX_LABEL_STYLE = {
    'display' : 'flex',
    'alignItems' : 'center',
    'gap' : '3px',
}
_INLINE_CONTROL_STYLE = {
    'display' : 'flex',
    'alignItems' : 'center',
    'gap' : '8px',
}


def _read_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def _path_state(path):
    if not path.exists():
        return {
            'exists' : False,
            'mtime_ns' : None,
            'size' : None,
        }
    stat = path.stat()
    return {
        'exists' : True,
        'mtime_ns' : str(stat.st_mtime_ns),
        'size' : stat.st_size,
    }


def _paths_state(paths):
    state = {}
    for path in paths:
        state[path.name] = _path_state(path)
    return state


def _status_item(name, exists):
    return html.Span([
        html.Span(
            style=_STATUS_DOT_STYLE if exists else _STATUS_DOT_MISSING_STYLE
        ),
        html.Span(f'{name}: {"present" if exists else "missing"}'),
    ], style=_STATUS_ITEM_STYLE)


def _last_updated_text(paths):
    timestamps = [path.stat().st_mtime for path in paths if path.exists()]
    if len(timestamps) == 0:
        return 'updated: --'
    return (
        'updated: '
        + pd.Timestamp(max(timestamps), unit='s').strftime('%Y-%m-%d %H:%M:%S')
    )


def _samples_controls():
    return html.Div([
        html.Div([
            html.Span('tag', style=_LABEL_STYLE),
            dcc.Dropdown(
                id='samples-tag',
                options=[
                    {'label' : 'raw', 'value' : 'raw'},
                    {'label' : 'standardized', 'value' : 'standardized'},
                ],
                value='raw',
                clearable=False,
                persistence=True,
                persistence_type='local',
                style={'fontSize' : '13px', 'width' : '100px'},
            ),
        ], style=_INLINE_CONTROL_STYLE),
        html.Div([
            html.Span('metric', style=_LABEL_STYLE),
            dcc.Dropdown(
                id='samples-metric',
                clearable=False,
                persistence=True,
                persistence_type='local',
                style={'fontSize' : '13px', 'width' : '100px'},
            ),
        ], style=_INLINE_CONTROL_STYLE),
        html.Div([
            html.Div([
                dcc.Checklist(
                    id='samples-auto-zoom-enabled',
                    options=[{'label' : '', 'value' : 'enabled'}],
                    value=[],
                    inline=True,
                    persistence=True,
                    persistence_type='local',
                    style={
                        'display' : 'flex',
                        'alignItems' : 'center',
                        'margin' : '0',
                    },
                    labelStyle={
                        'display' : 'inline-flex',
                        'alignItems' : 'center',
                        'margin' : '0',
                    },
                    inputStyle={
                        'margin' : '0',
                    },
                ),
                html.Span('auto zoom to last', style=_LABEL_STYLE),
            ], style=_CHECKBOX_LABEL_STYLE),
            dcc.Input(
                id='samples-auto-zoom-steps',
                type='number',
                min=1,
                step=1,
                value=10,
                disabled=True,
                persistence=True,
                persistence_type='local',
                style={
                    'height' : '22px',
                    'width' : '40px',
                    'padding' : '0',
                    'margin' : '0',
                    'border' : '1px solid #d0d7de',
                    'borderRadius' : '6px',
                    'backgroundColor' : '#ffffff',
                    'color' : '#2f3437',
                    'fontSize' : '13px',
                },
            ),
            html.Span('steps', style={'fontSize' : '12px', 'color' : '#4d545b'}),
        ], style=_CHECKBOX_ROW_STYLE),
    ], style=_CONTROLS_STYLE)


def _outcomes_controls():
    return html.Div([
        html.Div([
            html.Span('outcome', style=_LABEL_STYLE),
            dcc.Dropdown(
                id='outcomes-name',
                clearable=False,
                persistence=True,
                persistence_type='local',
                style={'fontSize' : '13px', 'width' : '280px'},
            ),
        ], style=_INLINE_CONTROL_STYLE),
        html.Div([
            html.Div([
                dcc.Checklist(
                    id='outcomes-auto-zoom-enabled',
                    options=[{'label' : '', 'value' : 'enabled'}],
                    value=[],
                    inline=True,
                    persistence=True,
                    persistence_type='local',
                    style={
                        'display' : 'flex',
                        'alignItems' : 'center',
                        'margin' : '0',
                    },
                    labelStyle={
                        'display' : 'inline-flex',
                        'alignItems' : 'center',
                        'margin' : '0',
                    },
                    inputStyle={
                        'margin' : '0',
                    },
                ),
                html.Span('auto zoom to last', style=_LABEL_STYLE),
            ], style=_CHECKBOX_LABEL_STYLE),
            dcc.Input(
                id='outcomes-auto-zoom-steps',
                type='number',
                min=1,
                step=1,
                value=10,
                disabled=True,
                persistence=True,
                persistence_type='local',
                style={
                    'height' : '32px',
                    'width' : '66px',
                    'padding' : '0 8px',
                    'border' : '1px solid #d0d7de',
                    'borderRadius' : '6px',
                    'backgroundColor' : '#ffffff',
                    'color' : '#2f3437',
                    'fontSize' : '13px',
                },
            ),
            html.Span('steps', style={'fontSize' : '12px', 'color' : '#4d545b'}),
        ], style=_CHECKBOX_ROW_STYLE),
    ], style=_CONTROLS_STYLE)


def _refresh_controls():
    return html.Div([
        html.Button(
            '⏸',
            id='refresh-toggle',
            n_clicks=0,
            title='pause refresh',
            style=_ICON_BUTTON_STYLE,
        ),
        html.Div([
            html.Span('refresh every', style=_LABEL_STYLE),
            dcc.Input(
                id='refresh-rate',
                type='number',
                min=0.1,
                step='any',
                value=1.0,
                persistence=True,
                persistence_type='local',
                style={
                    'height' : '22px',
                    'width' : '40px',
                    'padding' : '0',
                    'margin' : '5px 5px 5px 5px',
                    'border' : '1px solid #d0d7de',
                    'borderRadius' : '6px',
                    'backgroundColor' : '#ffffff',
                    'color' : '#2f3437',
                    'fontSize' : '13px',
                },
            ),
            html.Span('second(s)', style=_LABEL_STYLE),
        ]),
    ], style=_REFRESH_CONTROL_STYLE)


def _layout(log_dir, refresh_interval_ms=1000):
    log_dir = Path(log_dir)
    return html.Div([
        html.Div([_refresh_controls()], style=_UTILITY_BAR_STYLE),
        dcc.Interval(
            id='refresh-timer', interval=refresh_interval_ms, n_intervals=0, disabled=False
        ),
        dcc.Store(
            id='samples-files-state',
            data=_paths_state([log_dir / 'samples.csv']),
        ),
        dcc.Store(
            id='outcomes-files-state',
            data=_paths_state([log_dir / 'outcomes.csv', log_dir / 'targets.csv']),
        ),
        dcc.Store(id='samples-base-figure'),
        dcc.Store(id='outcomes-base-figure'),
        dcc.Store(id='samples-view-state'),
        dcc.Store(id='outcomes-view-state'),
        dcc.Tabs(id='view-tabs', value='samples', persistence=True, persistence_type='local',
                 children=[
            dcc.Tab(
                label='samples',
                value='samples',
                style=_TAB_STYLE,
                selected_style=_TAB_SELECTED_STYLE,
                children=[
                    html.Div([
                        _samples_controls(),
                        dcc.Graph(id='samples-graph', style=_GRAPH_STYLE),
                    ], style=_SECTION_STYLE)
                ],
            ),
            dcc.Tab(
                label='outcomes',
                value='outcomes',
                style=_TAB_STYLE,
                selected_style=_TAB_SELECTED_STYLE,
                children=[
                    html.Div([
                        _outcomes_controls(),
                        dcc.Graph(id='outcomes-graph', style=_GRAPH_STYLE),
                    ], style=_SECTION_STYLE)
                ],
            ),
        ], parent_style={'marginBottom' : '0'}, style=_TABS_STYLE, colors={
            'border' : '#d8dde3',
            'primary' : '#8c959f',
            'background' : '#f5f6f7',
        }),
        html.Div([
            html.Div(id='status-message', style=_STATUS_STYLE),
            html.Div(f'log dir: {log_dir}', style=_PATH_STYLE),
        ]),
    ], style=_PAGE_STYLE)


def _empty_figure(message):
    fig = go.Figure()
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font={'color' : '#2f3437'},
        margin={'l' : 48, 'r' : 24, 't' : 44, 'b' : 44},
        annotations=[{
            'text' : message,
            'xref' : 'paper',
            'yref' : 'paper',
            'x' : 0.5,
            'y' : 0.5,
            'showarrow' : False,
            'font' : {'size' : 14, 'color' : '#6c737a'},
        }],
    )
    return fig


def _apply_relayout(figure, relayout_data):
    if relayout_data is None:
        return
    axes = {}
    for key, value in relayout_data.items():
        if '.' not in key:
            continue
        axis_name, attribute = key.split('.', maxsplit=1)
        axes.setdefault(axis_name, {})[attribute] = value
    shared_xaxes = [
        axis_name
        for axis_name in axes
        if axis_name.startswith('xaxis')
        and getattr(getattr(figure.layout, axis_name, None), 'matches', None) == 'x'
    ]
    if shared_xaxes:
        xaxis_updates = next(
            (
                updates
                for axis_name, updates in axes.items()
                if axis_name in shared_xaxes and (
                    updates.get('autorange')
                    or (
                        updates.get('range[0]') is not None
                        and updates.get('range[1]') is not None
                    )
                )
            ),
            None,
        )
        if xaxis_updates is not None:
            if xaxis_updates.get('autorange'):
                figure.update_xaxes(autorange=True, range=None)
            else:
                figure.update_xaxes(
                    range=[
                        xaxis_updates['range[0]'],
                        xaxis_updates['range[1]'],
                    ],
                    autorange=False,
                )
            for axis_name in shared_xaxes:
                axes.pop(axis_name, None)
    for axis_name, updates in axes.items():
        axis = getattr(figure.layout, axis_name, None)
        if axis is None:
            continue
        if updates.get('autorange'):
            axis.autorange = True
            axis.range = None
            continue
        lower = updates.get('range[0]')
        upper = updates.get('range[1]')
        if lower is not None and upper is not None:
            axis.range = [lower, upper]
            axis.autorange = False


def _merge_relayout_state(current_state, relayout_data):
    if relayout_data is None:
        return current_state
    merged = {} if current_state is None else dict(current_state)
    for key, value in relayout_data.items():
        if '.' not in key:
            continue
        axis_name, attribute = key.split('.', maxsplit=1)
        if attribute == 'autorange' and value:
            merged[f'{axis_name}.autorange'] = True
            merged.pop(f'{axis_name}.range[0]', None)
            merged.pop(f'{axis_name}.range[1]', None)
            continue
        if attribute.startswith('range['):
            merged[f'{axis_name}.autorange'] = False
        merged[key] = value
    return merged


def _sample_color(step, min_step, max_step):
    if max_step <= min_step:
        position = 0.0
    else:
        position = (step - min_step) / (max_step - min_step)
    rgb = plotly.colors.sample_colorscale('Viridis', [position])[0]
    red, green, blue = [int(value) for value in rgb[4:-1].split(',')]
    return f'rgba({red}, {green}, {blue}, 0.35)'


def _integer_ticks(min_value, max_value):
    min_value = int(min_value)
    max_value = int(max_value)
    if max_value <= min_value:
        return [min_value, max_value]
    step_count = max_value - min_value
    if step_count <= 4:
        return list(range(min_value, max_value + 1))
    tick_count = min(5, step_count + 1)
    return sorted({
        min_value + round(i * step_count / (tick_count - 1))
        for i in range(tick_count)
    })


def _normalize_refresh_seconds(refresh_seconds):
    if refresh_seconds is None:
        return 1.0
    return max(float(refresh_seconds), 0.1)


def _normalize_step_count(step_count):
    if step_count is None:
        return 10
    return max(int(step_count), 1)


def _auto_zoom_enabled(value):
    return value is not None and 'enabled' in value


def _select_last_steps(data, step_count):
    if 'step' not in data.columns or len(data) == 0:
        return data
    steps = sorted(pd.Series(data['step']).dropna().unique())
    if len(steps) == 0:
        return data
    selected_steps = steps[-_normalize_step_count(step_count):]
    return data[data['step'].isin(selected_steps)]


def _numeric_range(values):
    values = pd.Series(values).dropna()
    if len(values) == 0:
        return None
    lower = float(values.min())
    upper = float(values.max())
    if lower == upper:
        padding = max(abs(lower) * 0.05, 1e-6)
    else:
        padding = (upper - lower) * 0.05
    return [lower - padding, upper + padding]


def _datetime_range(values):
    values = pd.to_datetime(pd.Series(values).dropna())
    if len(values) == 0:
        return None
    lower = values.min()
    upper = values.max()
    if lower == upper:
        padding = pd.Timedelta(seconds=1)
    else:
        padding = (upper - lower) / 20
    return [lower - padding, upper + padding]


def _default_samples_metric(samples, tag, current_value):
    tagged = samples[samples['tag'] == tag]
    metrics = [col for col in tagged.columns if col.startswith('metric_')]
    if not metrics:
        return None, []
    if current_value in metrics:
        return current_value, metrics
    return metrics[0], metrics


def _default_outcome_name(outcomes, current_value):
    outcome_names = list(outcomes['outcome_name'].drop_duplicates())
    if not outcome_names:
        return None, []
    if current_value in outcome_names:
        return current_value, outcome_names
    return outcome_names[0], outcome_names


def _samples_figure(samples, tag, metric):
    tagged = samples[samples['tag'] == tag].copy()
    params = [col for col in tagged.columns if col.startswith('param_')]
    if len(tagged) == 0 or metric is None or len(params) == 0:
        return _empty_figure('no samples')

    rows = max(1, int(len(params) ** 0.5))
    cols = (len(params) + rows - 1) // rows
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[param[6:] for param in params],
        shared_yaxes=True,
        shared_xaxes=(tag == 'standardized'),
    )
    for position, param in enumerate(params):
        row = position // cols + 1
        col = position % cols + 1
        hovertemplate = (
            f'{param[6:]}=%{{x}}<br>{metric[7:]}=%{{y}}<br>step=%{{marker.color}}<extra></extra>'
        )
        fig.add_trace(
            go.Scattergl(
                x=tagged[param].tolist(),
                y=tagged[metric].tolist(),
                mode='markers',
                marker={
                    'symbol' : 'cross',
                    'size' : 5,
                    'color' : tagged['step'].tolist(),
                    'coloraxis' : 'coloraxis',
                },
                showlegend=False,
                hovertemplate=hovertemplate,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text=param[6:], row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text=metric[7:], row=row, col=col)
    if tag == 'standardized':
        fig.update_xaxes(matches='x')

    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font={'color' : '#2f3437'},
        margin={'l' : 56, 'r' : 24, 't' : 52, 'b' : 44},
        title=f'samples / {metric[7:]}',
        coloraxis={
            'colorscale' : 'Viridis',
            'cmin' : tagged['step'].min(),
            'cmax' : max(tagged['step'].min() + 1, tagged['step'].max()),
            'colorbar' : {
                'title' : {'text' : 'step', 'font' : {'size' : 12}},
                'tickfont' : {'size' : 11},
                'thickness' : 14,
                'len' : 0.82,
                'tickmode' : 'array',
                'tickvals' : _integer_ticks(
                    tagged['step'].min(),
                    max(tagged['step'].min() + 1, tagged['step'].max()),
                ),
            },
        },
        uirevision=f'samples:{tag}:{metric}',
    )
    return fig


def _outcomes_figure(outcomes, targets, outcome_name):
    selected = outcomes[outcomes['outcome_name'] == outcome_name].copy()
    if len(selected) == 0:
        return _empty_figure('no outcomes')

    selected['time'] = pd.to_datetime(selected['time'])
    grouped = list(selected.groupby(['step', 'index'], sort=False))
    steps = sorted({int(step) for (step, _), _ in grouped})
    min_step = min(steps)
    max_step = max(steps)

    fig = go.Figure()
    for (step, index), group in grouped:
        fig.add_trace(go.Scattergl(
            x=group['time'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f').tolist(),
            y=group['predicted_value'].tolist(),
            mode='lines',
            line={
                'color' : _sample_color(int(step), min_step, max_step),
                'width' : 1.5,
            },
            meta={'kind' : 'prediction', 'step' : int(step), 'index' : int(index)},
            name=f'step {step}, index {index}',
            showlegend=False,
            hovertemplate=(
                f'step={step}<br>index={index}<br>time=%{{x}}<br>predicted=%{{y}}<extra></extra>'
            ),
        ))
    fig.add_trace(go.Scatter(
        x=[selected['time'].iloc[0].strftime('%Y-%m-%dT%H:%M:%S.%f')],
        y=[float(selected['predicted_value'].iloc[0])],
        mode='markers',
        marker={
            'size' : 0.1,
            'opacity' : 0,
            'color' : [min_step],
            'coloraxis' : 'coloraxis',
        },
        meta={'kind' : 'colorbar'},
        showlegend=False,
        hoverinfo='skip',
    ))

    if targets is not None and 'outcome_name' in targets.columns:
        target_data = targets[targets['outcome_name'] == outcome_name].copy()
        if len(target_data) > 0:
            target_data['time'] = pd.to_datetime(target_data['time'])
            fig.add_trace(go.Scatter(
                x=target_data['time'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f').tolist(),
                y=target_data['target_value'].tolist(),
                mode='markers',
                marker={'color' : 'red', 'size' : 8},
                meta={'kind' : 'target'},
                name='target',
                showlegend=True,
                hovertemplate='target<br>time=%{x}<br>value=%{y}<extra></extra>',
            ))

    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font={'color' : '#2f3437'},
        margin={'l' : 56, 'r' : 24, 't' : 52, 'b' : 44},
        title=f'outcomes / {outcome_name}',
        coloraxis={
            'colorscale' : 'Viridis',
            'cmin' : min_step,
            'cmax' : max(min_step + 1, max_step),
            'colorbar' : {
                'title' : {'text' : 'step', 'font' : {'size' : 12}},
                'tickfont' : {'size' : 11},
                'thickness' : 14,
                'len' : 0.82,
                'tickmode' : 'array',
                'tickvals' : _integer_ticks(min_step, max(min_step + 1, max_step)),
            },
        },
        xaxis_title='time',
        yaxis_title='predicted value',
        legend={
            'x' : 1.0,
            'xanchor' : 'right',
            'y' : 1.0,
            'bgcolor' : 'rgba(255,255,255,0.85)',
            'bordercolor' : '#d8dde3',
            'borderwidth' : 1,
            'font' : {'size' : 11, 'color' : '#57606a'},
        },
        uirevision=f'outcomes:{outcome_name}',
    )
    return fig


def create_app(log_dir, refresh_interval_ms):
    # pylint: disable=too-many-locals
    '''Create the Dash app instance.'''
    log_dir = Path(log_dir)
    samples_path = log_dir / 'samples.csv'
    outcomes_path = log_dir / 'outcomes.csv'
    targets_path = log_dir / 'targets.csv'

    app = Dash(__name__, update_title=None)
    app.title = 'Daisy calibration monitor'
    app.layout = _layout(log_dir, refresh_interval_ms)

    @app.callback(
        Output('samples-files-state', 'data'),
        Output('outcomes-files-state', 'data'),
        Input('refresh-timer', 'n_intervals'),
        State('samples-files-state', 'data'),
        State('outcomes-files-state', 'data'),
    )
    def update_files_state(_, current_samples_state, current_outcomes_state):
        samples_state = _paths_state([samples_path])
        outcomes_state = _paths_state([outcomes_path, targets_path])
        return (
            no_update if samples_state == current_samples_state else samples_state,
            no_update if outcomes_state == current_outcomes_state else outcomes_state,
        )

    @app.callback(
        Output('refresh-timer', 'interval'),
        Input('refresh-rate', 'value'),
    )
    def update_refresh_rate(refresh_seconds):
        return int(_normalize_refresh_seconds(refresh_seconds) * 1000)

    @app.callback(
        Output('samples-auto-zoom-steps', 'disabled'),
        Input('samples-auto-zoom-enabled', 'value'),
    )
    def update_samples_auto_zoom_disabled(auto_zoom):
        return not _auto_zoom_enabled(auto_zoom)

    @app.callback(
        Output('outcomes-auto-zoom-steps', 'disabled'),
        Input('outcomes-auto-zoom-enabled', 'value'),
    )
    def update_outcomes_auto_zoom_disabled(auto_zoom):
        return not _auto_zoom_enabled(auto_zoom)

    @app.callback(
        Output('refresh-timer', 'disabled'),
        Output('refresh-toggle', 'children'),
        Output('refresh-toggle', 'title'),
        Input('refresh-toggle', 'n_clicks'),
    )
    def toggle_refresh(n_clicks):
        disabled = n_clicks % 2 == 1
        if disabled:
            return True, '▶', 'start refresh'
        return False, '⏸', 'pause refresh'

    @app.callback(
        Output('status-message', 'children'),
        Input('refresh-timer', 'n_intervals'),
        Input('refresh-rate', 'value'),
        Input('refresh-timer', 'disabled'),
    )
    def update_status(_, refresh_seconds, refresh_disabled):
        refresh_seconds = _normalize_refresh_seconds(refresh_seconds)
        return [
            _status_item('samples', samples_path.exists()),
            _status_item('outcomes', outcomes_path.exists()),
            _status_item('targets', targets_path.exists()),
            html.Span(
                f'refresh: {"stopped" if refresh_disabled else f"{refresh_seconds:g} s"}'
            ),
            html.Span(_last_updated_text([samples_path, outcomes_path, targets_path])),
        ]

    @app.callback(
        Output('samples-metric', 'options'),
        Output('samples-metric', 'value'),
        Input('view-tabs', 'value'),
        Input('samples-files-state', 'data'),
        Input('samples-tag', 'value'),
        State('samples-metric', 'value'),
    )
    def update_samples_metric(active_tab, _, tag, current_metric):
        if active_tab != 'samples':
            return no_update, no_update
        samples = _read_csv(samples_path)
        if samples is None or 'tag' not in samples.columns:
            return [], None
        metric, metrics = _default_samples_metric(samples, tag, current_metric)
        return [{'label' : name[7:], 'value' : name} for name in metrics], metric

    @app.callback(
        Output('samples-view-state', 'data'),
        Input('samples-graph', 'relayoutData'),
        State('samples-view-state', 'data'),
    )
    def update_samples_view_state(relayout_data, current_state):
        return _merge_relayout_state(current_state, relayout_data)

    @app.callback(
        Output('samples-base-figure', 'data'),
        Input('view-tabs', 'value'),
        Input('samples-files-state', 'data'),
        Input('samples-tag', 'value'),
        Input('samples-metric', 'value'),
        State('samples-view-state', 'data'),
    )
    def update_samples_graph(active_tab, _, tag, metric, relayout_data):
        if active_tab != 'samples':
            return no_update
        samples = _read_csv(samples_path)
        if samples is None:
            return _empty_figure('no samples').to_plotly_json()
        if 'tag' not in samples.columns:
            return _empty_figure('invalid samples').to_plotly_json()
        metric, _ = _default_samples_metric(samples, tag, metric)
        figure = _samples_figure(samples, tag, metric)
        _apply_relayout(figure, relayout_data)
        return figure.to_plotly_json()

    @app.callback(
        Output('outcomes-name', 'options'),
        Output('outcomes-name', 'value'),
        Input('view-tabs', 'value'),
        Input('outcomes-files-state', 'data'),
        State('outcomes-name', 'value'),
    )
    def update_outcomes_name(active_tab, _, current_value):
        if active_tab != 'outcomes':
            return no_update, no_update
        outcomes = _read_csv(outcomes_path)
        if outcomes is None or 'outcome_name' not in outcomes.columns:
            return [], None
        outcome_name, outcome_names = _default_outcome_name(outcomes, current_value)
        return [{'label' : name, 'value' : name} for name in outcome_names], outcome_name

    @app.callback(
        Output('outcomes-view-state', 'data'),
        Input('outcomes-graph', 'relayoutData'),
        State('outcomes-view-state', 'data'),
    )
    def update_outcomes_view_state(relayout_data, current_state):
        return _merge_relayout_state(current_state, relayout_data)

    @app.callback(
        Output('outcomes-base-figure', 'data'),
        Input('view-tabs', 'value'),
        Input('outcomes-files-state', 'data'),
        Input('outcomes-name', 'value'),
        State('outcomes-view-state', 'data'),
    )
    def update_outcomes_graph(active_tab, _, outcome_name, relayout_data):
        if active_tab != 'outcomes':
            return no_update
        outcomes = _read_csv(outcomes_path)
        if outcomes is None:
            return _empty_figure('no outcomes').to_plotly_json()
        if 'outcome_name' not in outcomes.columns:
            return _empty_figure('invalid outcomes').to_plotly_json()
        outcome_name, _ = _default_outcome_name(outcomes, outcome_name)
        if outcome_name is None:
            return _empty_figure('no outcomes').to_plotly_json()
        targets = _read_csv(targets_path)
        figure = _outcomes_figure(outcomes, targets, outcome_name)
        _apply_relayout(figure, relayout_data)
        return figure.to_plotly_json()

    # pylint: disable=line-too-long
    app.clientside_callback(
        '''
        function(baseFigure, autoZoomValue, autoZoomSteps) {
            if (!baseFigure) {
                return window.dash_clientside.no_update;
            }

            const enabled = Array.isArray(autoZoomValue) && autoZoomValue.includes('enabled');
            if (!enabled) {
                return baseFigure;
            }

            const stepCount = Math.max(parseInt(autoZoomSteps ?? 10, 10) || 10, 1);
            const figure = JSON.parse(JSON.stringify(baseFigure));
            const traces = Array.isArray(figure.data) ? figure.data : [];
            if (traces.length === 0) {
                return figure;
            }

            const stepValues = (((traces[0] || {}).marker || {}).color || []).filter(
                value => value !== null && value !== undefined
            );
            const steps = Array.from(new Set(stepValues)).sort((a, b) => a - b);
            if (steps.length === 0) {
                return figure;
            }
            const selectedSteps = new Set(steps.slice(-stepCount));

            function numericRange(values) {
                const filtered = values.filter(value => Number.isFinite(value));
                if (filtered.length === 0) {
                    return null;
                }
                const lower = Math.min(...filtered);
                const upper = Math.max(...filtered);
                const padding = lower === upper
                    ? Math.max(Math.abs(lower) * 0.05, 1e-6)
                    : (upper - lower) * 0.05;
                return [lower - padding, upper + padding];
            }

            function axisKey(axisRef, axisPrefix) {
                if (!axisRef || axisRef === axisPrefix) {
                    return axisPrefix + 'axis';
                }
                return axisPrefix + 'axis' + axisRef.slice(axisPrefix.length);
            }

            const yValues = [];
            const xByAxis = {};
            for (const trace of traces) {
                const colors = (((trace || {}).marker || {}).color || []);
                const xs = trace.x || [];
                const ys = trace.y || [];
                const key = axisKey(trace.xaxis || 'x', 'x');
                if (!(key in xByAxis)) {
                    xByAxis[key] = [];
                }
                for (let index = 0; index < Math.min(colors.length, xs.length, ys.length); index += 1) {
                    if (!selectedSteps.has(colors[index])) {
                        continue;
                    }
                    if (Number.isFinite(ys[index])) {
                        yValues.push(ys[index]);
                    }
                    if (Number.isFinite(xs[index])) {
                        xByAxis[key].push(xs[index]);
                    }
                }
            }

            const yRange = numericRange(yValues);
            if (yRange) {
                for (const [key, value] of Object.entries(figure.layout || {})) {
                    if (key.startsWith('yaxis')) {
                        value.range = yRange;
                        value.autorange = false;
                    }
                }
            }

            const sharedX = Object.entries(figure.layout || {}).some(
                ([key, value]) => key.startsWith('xaxis') && value && value.matches === 'x'
            );
            if (sharedX) {
                const sharedValues = Object.values(xByAxis).flat();
                const xRange = numericRange(sharedValues);
                if (xRange) {
                    for (const [key, value] of Object.entries(figure.layout || {})) {
                        if (key.startsWith('xaxis')) {
                            value.range = xRange;
                            value.autorange = false;
                        }
                    }
                }
            } else {
                for (const [key, values] of Object.entries(xByAxis)) {
                    const xRange = numericRange(values);
                    if (xRange && figure.layout && figure.layout[key]) {
                        figure.layout[key].range = xRange;
                        figure.layout[key].autorange = false;
                    }
                }
            }

            return figure;
        }
        ''',
        Output('samples-graph', 'figure'),
        Input('samples-base-figure', 'data'),
        Input('samples-auto-zoom-enabled', 'value'),
        Input('samples-auto-zoom-steps', 'value'),
    )
    # pylint: enable=line-too-long

    app.clientside_callback(
        '''
        function(baseFigure, autoZoomValue, autoZoomSteps) {
            if (!baseFigure) {
                return window.dash_clientside.no_update;
            }

            const enabled = Array.isArray(autoZoomValue) && autoZoomValue.includes('enabled');
            if (!enabled) {
                return baseFigure;
            }

            const stepCount = Math.max(parseInt(autoZoomSteps ?? 10, 10) || 10, 1);
            const figure = JSON.parse(JSON.stringify(baseFigure));
            const traces = Array.isArray(figure.data) ? figure.data : [];
            const selectedTraces = traces.filter(
                trace => trace.meta && trace.meta.kind === 'prediction'
            );
            if (selectedTraces.length === 0) {
                return figure;
            }

            const steps = Array.from(new Set(selectedTraces.map(trace => trace.meta.step))).sort(
                (a, b) => a - b
            );
            const selectedSteps = new Set(steps.slice(-stepCount));

            function numericRange(values) {
                const filtered = values.filter(value => Number.isFinite(value));
                if (filtered.length === 0) {
                    return null;
                }
                const lower = Math.min(...filtered);
                const upper = Math.max(...filtered);
                const padding = lower === upper
                    ? Math.max(Math.abs(lower) * 0.05, 1e-6)
                    : (upper - lower) * 0.05;
                return [lower - padding, upper + padding];
            }

            function datetimeRange(values) {
                const filtered = values
                    .map(value => Date.parse(value))
                    .filter(value => Number.isFinite(value));
                if (filtered.length === 0) {
                    return null;
                }
                const lower = Math.min(...filtered);
                const upper = Math.max(...filtered);
                const padding = lower === upper ? 1000 : (upper - lower) / 20;
                return [
                    new Date(lower - padding).toISOString(),
                    new Date(upper + padding).toISOString(),
                ];
            }

            const xs = [];
            const ys = [];
            for (const trace of selectedTraces) {
                if (!selectedSteps.has(trace.meta.step)) {
                    continue;
                }
                for (const value of (trace.x || [])) {
                    xs.push(value);
                }
                for (const value of (trace.y || [])) {
                    ys.push(value);
                }
            }

            const xRange = datetimeRange(xs);
            if (xRange && figure.layout && figure.layout.xaxis) {
                figure.layout.xaxis.range = xRange;
                figure.layout.xaxis.autorange = false;
            }

            const yRange = numericRange(ys);
            if (yRange && figure.layout && figure.layout.yaxis) {
                figure.layout.yaxis.range = yRange;
                figure.layout.yaxis.autorange = false;
            }

            return figure;
        }
        ''',
        Output('outcomes-graph', 'figure'),
        Input('outcomes-base-figure', 'data'),
        Input('outcomes-auto-zoom-enabled', 'value'),
        Input('outcomes-auto-zoom-steps', 'value'),
    )

    return app


def run_app(app, host, port, *, open_browser=True, verbose=False):
    '''Run the web app with quiet defaults suitable for end users.'''
    url = f'http://{host}:{port}'
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    print(f'Monitor available at {url}',
          '',
          'To stop the monitor:',
          '    Press Ctrl+C in this terminal.',
          '',
          'Leave this terminal window open while using the monitor.',
          sep='\n')
    if not verbose:
        logging.getLogger('waitress').setLevel(logging.ERROR)
        app.server.logger.setLevel(logging.ERROR)

    serve(
        app.server,
        host=host,
        port=port,
        _quiet=not verbose,
    )
