'''Local web app for monitoring optimization logs.'''
import argparse
import logging
import threading
import webbrowser
from pathlib import Path

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from waitress import serve


def _sanitize_name(name):
    return ''.join(c if c.isalnum() or c in ('-', '_') else '-' for c in name)


def _read_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def _samples_controls():
    return html.Div([
        html.Div([
            html.Label('Tag'),
            dcc.Dropdown(
                id='samples-tag',
                options=[
                    {'label' : 'raw', 'value' : 'raw'},
                    {'label' : 'standardized', 'value' : 'standardized'},
                ],
                value='raw',
                clearable=False,
            ),
        ], style={'width' : '220px'}),
        html.Div([
            html.Label('Metric'),
            dcc.Dropdown(id='samples-metric', clearable=False),
        ], style={'width' : '320px'}),
        html.Button('Reset to current data', id='samples-reset', n_clicks=0),
    ], style={
        'display' : 'flex',
        'gap' : '1rem',
        'alignItems' : 'end',
        'flexWrap' : 'wrap',
        'marginBottom' : '1rem',
    })


def _outcomes_controls():
    return html.Div([
        html.Div([
            html.Label('Outcome'),
            dcc.Dropdown(id='outcomes-name', clearable=False),
        ], style={'width' : '320px'}),
        html.Button('Reset to current data', id='outcomes-reset', n_clicks=0),
    ], style={
        'display' : 'flex',
        'gap' : '1rem',
        'alignItems' : 'end',
        'flexWrap' : 'wrap',
        'marginBottom' : '1rem',
    })


def _layout():
    return html.Div([
        html.H1('daisypy-optim monitor'),
        html.Div(id='status-message', style={'marginBottom' : '1rem'}),
        dcc.Interval(id='refresh-timer', interval=1000, n_intervals=0),
        dcc.Store(id='samples-view-state'),
        dcc.Store(id='outcomes-view-state'),
        dcc.Tabs(id='view-tabs', value='samples', children=[
            dcc.Tab(
                label='Samples',
                value='samples',
                children=[
                    html.Div([
                        _samples_controls(),
                        dcc.Graph(id='samples-graph', style={'height' : '85vh'}),
                    ], style={'paddingTop' : '1rem'})
                ],
            ),
            dcc.Tab(
                label='Outcomes',
                value='outcomes',
                children=[
                    html.Div([
                        _outcomes_controls(),
                        dcc.Graph(id='outcomes-graph', style={'height' : '85vh'}),
                    ], style={'paddingTop' : '1rem'})
                ],
            ),
        ]),
    ], style={'padding' : '1rem 1.5rem'})


def _empty_figure(message):
    fig = go.Figure()
    fig.update_layout(
        template='plotly_white',
        annotations=[{
            'text' : message,
            'xref' : 'paper',
            'yref' : 'paper',
            'x' : 0.5,
            'y' : 0.5,
            'showarrow' : False,
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
    return list(range(int(min_value), int(max_value) + 1))


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


def _samples_figure(samples, tag, metric, reset_count):
    tagged = samples[samples['tag'] == tag].copy()
    params = [col for col in tagged.columns if col.startswith('param_')]
    if len(tagged) == 0 or metric is None or len(params) == 0:
        return _empty_figure(f'No sample data for tag "{tag}"')

    rows = max(1, int(len(params) ** 0.5))
    cols = (len(params) + rows - 1) // rows
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[param[6:] for param in params],
        shared_yaxes=True,
        shared_xaxes=False,
    )
    for position, param in enumerate(params):
        row = position // cols + 1
        col = position % cols + 1
        fig.add_trace(
            go.Scattergl(
                x=tagged[param],
                y=tagged[metric],
                mode='markers',
                marker={
                    'symbol' : 'cross',
                    'size' : 10,
                    'color' : tagged['step'],
                    'coloraxis' : 'coloraxis',
                },
                showlegend=False,
                hovertemplate=(
                    f'{param[6:]}=%{{x}}<br>{metric[7:]}=%{{y}}<br>step=%{{marker.color}}<extra></extra>'
                ),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text=param[6:], row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text=metric[7:], row=row, col=col)

    fig.update_layout(
        template='plotly_white',
        title=f'Marginal distribution of sampled parameters vs Objective ({metric[7:]})',
        coloraxis={
            'colorscale' : 'Viridis',
            'cmin' : tagged['step'].min(),
            'cmax' : max(tagged['step'].min() + 1, tagged['step'].max()),
            'colorbar' : {
                'title' : 'run',
                'tickmode' : 'array',
                'tickvals' : _integer_ticks(
                    tagged['step'].min(),
                    max(tagged['step'].min() + 1, tagged['step'].max()),
                ),
            },
        },
        uirevision=f'samples:{tag}:{metric}:{reset_count}',
    )
    return fig


def _outcomes_figure(outcomes, targets, outcome_name, reset_count):
    selected = outcomes[outcomes['outcome_name'] == outcome_name].copy()
    if len(selected) == 0:
        return _empty_figure(f'No outcome data for "{outcome_name}"')

    selected['time'] = pd.to_datetime(selected['time'])
    grouped = list(selected.groupby(['step', 'index'], sort=False))
    steps = sorted({int(step) for (step, _), _ in grouped})
    min_step = min(steps)
    max_step = max(steps)

    fig = go.Figure()
    for (step, index), group in grouped:
        fig.add_trace(go.Scatter(
            x=group['time'],
            y=group['predicted_value'],
            mode='lines',
            line={
                'color' : _sample_color(int(step), min_step, max_step),
                'width' : 1.5,
            },
            name=f'step {step}, index {index}',
            showlegend=False,
            hovertemplate=(
                f'step={step}<br>index={index}<br>time=%{{x}}<br>predicted=%{{y}}<extra></extra>'
            ),
        ))

    if targets is not None and 'outcome_name' in targets.columns:
        target_data = targets[targets['outcome_name'] == outcome_name].copy()
        if len(target_data) > 0:
            target_data['time'] = pd.to_datetime(target_data['time'])
            fig.add_trace(go.Scatter(
                x=target_data['time'],
                y=target_data['target_value'],
                mode='markers',
                marker={'color' : 'red', 'size' : 8},
                name='target',
                hovertemplate='target<br>time=%{x}<br>value=%{y}<extra></extra>',
            ))

    fig.update_layout(
        template='plotly_white',
        title=f'{outcome_name} outcome curves ({len(grouped)} evaluations)',
        xaxis_title='time',
        yaxis_title='predicted value',
        uirevision=f'outcomes:{outcome_name}:{reset_count}',
    )
    return fig


def create_app(log_dir, refresh_interval_ms):
    '''Create the Dash app instance.'''
    log_dir = Path(log_dir)
    samples_path = log_dir / 'samples.csv'
    outcomes_path = log_dir / 'outcomes.csv'
    targets_path = log_dir / 'targets.csv'

    app = Dash(__name__, update_title=None)
    app.title = 'daisypy-optim monitor'
    app.layout = _layout()
    app.layout.children[2].interval = refresh_interval_ms

    @app.callback(
        Output('status-message', 'children'),
        Input('refresh-timer', 'n_intervals'),
    )
    def update_status(_):
        messages = [f'Log directory: {log_dir}']
        messages.append(
            f'samples.csv: {"found" if samples_path.exists() else "missing"}'
        )
        messages.append(
            f'outcomes.csv: {"found" if outcomes_path.exists() else "missing"}'
        )
        messages.append(
            f'targets.csv: {"found" if targets_path.exists() else "missing"}'
        )
        return ' | '.join(messages)

    @app.callback(
        Output('samples-metric', 'options'),
        Output('samples-metric', 'value'),
        Input('refresh-timer', 'n_intervals'),
        Input('samples-tag', 'value'),
        State('samples-metric', 'value'),
    )
    def update_samples_metric(_, tag, current_metric):
        samples = _read_csv(samples_path)
        if samples is None or 'tag' not in samples.columns:
            return [], None
        metric, metrics = _default_samples_metric(samples, tag, current_metric)
        return [{'label' : name[7:], 'value' : name} for name in metrics], metric

    @app.callback(
        Output('samples-view-state', 'data'),
        Input('samples-graph', 'relayoutData'),
        Input('samples-reset', 'n_clicks'),
        State('samples-view-state', 'data'),
    )
    def update_samples_view_state(relayout_data, _, current_state):
        if ctx.triggered_id == 'samples-reset':
            return None
        if ctx.triggered_id == 'samples-graph':
            return _merge_relayout_state(current_state, relayout_data)
        return no_update

    @app.callback(
        Output('samples-graph', 'figure'),
        Input('refresh-timer', 'n_intervals'),
        Input('samples-tag', 'value'),
        Input('samples-metric', 'value'),
        Input('samples-reset', 'n_clicks'),
        Input('samples-view-state', 'data'),
    )
    def update_samples_graph(_, tag, metric, reset_count, relayout_data):
        samples = _read_csv(samples_path)
        if samples is None:
            return _empty_figure(f'Missing {samples_path.name}')
        if 'tag' not in samples.columns:
            return _empty_figure(f'{samples_path.name} must contain a tag column')
        figure = _samples_figure(samples, tag, metric, reset_count)
        if ctx.triggered_id != 'samples-reset':
            _apply_relayout(figure, relayout_data)
        return figure

    @app.callback(
        Output('outcomes-name', 'options'),
        Output('outcomes-name', 'value'),
        Input('refresh-timer', 'n_intervals'),
        State('outcomes-name', 'value'),
    )
    def update_outcomes_name(_, current_value):
        outcomes = _read_csv(outcomes_path)
        if outcomes is None or 'outcome_name' not in outcomes.columns:
            return [], None
        outcome_name, outcome_names = _default_outcome_name(outcomes, current_value)
        return [{'label' : name, 'value' : name} for name in outcome_names], outcome_name

    @app.callback(
        Output('outcomes-view-state', 'data'),
        Input('outcomes-graph', 'relayoutData'),
        Input('outcomes-reset', 'n_clicks'),
        State('outcomes-view-state', 'data'),
    )
    def update_outcomes_view_state(relayout_data, _, current_state):
        if ctx.triggered_id == 'outcomes-reset':
            return None
        if ctx.triggered_id == 'outcomes-graph':
            return _merge_relayout_state(current_state, relayout_data)
        return no_update

    @app.callback(
        Output('outcomes-graph', 'figure'),
        Input('refresh-timer', 'n_intervals'),
        Input('outcomes-name', 'value'),
        Input('outcomes-reset', 'n_clicks'),
        Input('outcomes-view-state', 'data'),
    )
    def update_outcomes_graph(_, outcome_name, reset_count, relayout_data):
        outcomes = _read_csv(outcomes_path)
        if outcomes is None:
            return _empty_figure(f'Missing {outcomes_path.name}')
        if 'outcome_name' not in outcomes.columns:
            return _empty_figure(f'{outcomes_path.name} must contain an outcome_name column')
        if outcome_name is None:
            return _empty_figure(f'No outcomes found in {outcomes_path.name}')
        targets = _read_csv(targets_path)
        figure = _outcomes_figure(outcomes, targets, outcome_name, reset_count)
        if ctx.triggered_id != 'outcomes-reset':
            _apply_relayout(figure, relayout_data)
        return figure

    return app


def run_app(app, host, port, *, open_browser=True, verbose=False):
    '''Run the web app with quiet defaults suitable for end users.'''
    url = f'http://{host}:{port}'
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    print(f'Monitor available at {url}')
    if not verbose:
        logging.getLogger('waitress').setLevel(logging.ERROR)
        app.server.logger.setLevel(logging.ERROR)
    serve(
        app.server,
        host=host,
        port=port,
        _quiet=not verbose,
    )


def main():
    '''Run the local web app.'''
    parser = argparse.ArgumentParser(
        description='Run a local web app for monitoring optimization logs'
    )
    parser.add_argument(
        'log_dir',
        type=Path,
        help='Directory containing samples.csv and optionally outcomes.csv and targets.csv',
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host interface to bind.',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Port to bind.',
    )
    parser.add_argument(
        '--refresh-seconds',
        type=float,
        default=1.0,
        help='Seconds between file refreshes.',
    )
    parser.add_argument(
        '--no-open-browser',
        action='store_true',
        default=False,
        help='Do not open the browser automatically.',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Enable server request logging.',
    )
    args = parser.parse_args()

    app = create_app(args.log_dir, int(max(args.refresh_seconds, 0.1) * 1000))
    run_app(
        app,
        args.host,
        args.port,
        open_browser=not args.no_open_browser,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
