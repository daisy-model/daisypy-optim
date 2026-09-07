# pylint: disable=R0801
'''Manual interactive test for the monitor samples view.'''
import argparse
import sys
import threading
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from daisypy.optim.csv_log import CsvLog
from daisypy.optim.monitor import create_app, run_app
# pylint: enable=wrong-import-position


def _make_standardized(values, centre, scale):
    standardized = (values - centre) / scale
    return np.clip(standardized, -1.0, 1.0)


def _generate_step_samples(rng, centre, scale, optimum):
    centre = centre + 0.25 * (optimum - centre) + rng.normal(0.0, 0.08 * scale)
    scale = np.maximum(0.08, scale * rng.uniform(0.94, 0.99, size=2))
    num_samples = int(rng.integers(4, 10))
    samples = centre + rng.normal(size=(num_samples, 2)) * scale
    losses = (
        ((samples[:, 0] - optimum[0]) / 1.4) ** 2
        + ((samples[:, 1] - optimum[1]) / 0.55) ** 2
        + rng.uniform(0.0, 0.15, size=num_samples)
    )
    centre = 0.6 * centre + 0.4 * samples[np.argmin(losses)]
    return samples, losses, centre, scale


def _write_outcome(log, step, index, loss, sample):
    times = [
        '2000-01-01T00:00:00',
        '2000-01-02T00:00:00',
        '2000-01-03T00:00:00',
    ]
    baseline = 10.0 - loss
    for day, time in enumerate(times):
        log.log(
            step=step,
            index=index,
            outcome_name='mock-outcome',
            time=time,
            predicted_value=baseline + 0.2 * day + 0.05 * sample[0] - 0.03 * sample[1],
        )


def _write_targets(log):
    for day, time in enumerate([
            '2000-01-01T00:00:00',
            '2000-01-02T00:00:00',
            '2000-01-03T00:00:00',
    ]):
        log.log(
            objective_name='mock-objective',
            outcome_name='mock-outcome',
            time=time,
            target_value=9.5 + 0.2 * day,
        )


def _write_live_samples(log_dir, write_interval, ready_event, stop_event):
    # pylint: disable=too-many-locals
    samples_path = Path(log_dir) / 'samples.csv'
    outcomes_path = Path(log_dir) / 'outcomes.csv'
    targets_path = Path(log_dir) / 'targets.csv'
    sample_columns = [
        'step',
        'index',
        'tag',
        'metric_loss',
        'param_x',
        'param_y',
    ]
    outcome_columns = [
        'step',
        'index',
        'outcome_name',
        'time',
        'predicted_value',
    ]
    target_columns = [
        'objective_name',
        'outcome_name',
        'time',
        'target_value',
    ]
    rng = np.random.default_rng()
    optimum = np.array([2.0, -0.8], dtype=float)
    centre = optimum + rng.normal(loc=[3.5, 1.2], scale=[1.0, 0.5])
    scale = np.array([1.8, 0.7], dtype=float)

    with CsvLog(samples_path, columns=sample_columns) as sample_log, \
            CsvLog(outcomes_path, columns=outcome_columns) as outcome_log, \
            CsvLog(targets_path, columns=target_columns) as target_log:
        _write_targets(target_log)
        step = 0
        while not stop_event.is_set():
            samples, losses, centre, scale = _generate_step_samples(
                rng,
                centre,
                scale,
                optimum,
            )
            standardized = _make_standardized(samples, optimum, np.array([1.8, 0.7]))
            for index, (sample, sample_standardized, loss) in enumerate(
                    zip(samples, standardized, losses, strict=True)
            ):
                sample_log.log(
                    step=step,
                    index=index,
                    tag='raw',
                    metric_loss=loss,
                    param_x=sample[0],
                    param_y=sample[1],
                )
                sample_log.log(
                    step=step,
                    index=index,
                    tag='standardized',
                    metric_loss=loss,
                    param_x=sample_standardized[0],
                    param_y=sample_standardized[1],
                )
                _write_outcome(outcome_log, step, index, loss, sample)
            if step == 0:
                ready_event.set()
            step += 1
            if stop_event.wait(write_interval):
                break


def _writer_entrypoint(args, ready_event, stop_event, failure):
    try:
        _write_live_samples(
            args.log_dir,
            args.write_interval,
            ready_event,
            stop_event,
        )
    except Exception as exc: # pylint: disable=broad-exception-caught
        failure.append(exc)
        ready_event.set()


def main():
    '''Run a manual live web app scenario for the samples view.'''
    parser = argparse.ArgumentParser(
        description='Manual interactive test for the monitor samples view'
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path('manual-tests/out/live-visualize-samples'),
        help='Directory where samples.csv is written.',
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
        help='Seconds between browser refreshes.',
    )
    parser.add_argument(
        '--write-interval',
        type=float,
        default=2.0,
        help='Seconds between new optimization-like sample batches.',
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

    args.log_dir.mkdir(parents=True, exist_ok=True)
    ready_event = threading.Event()
    stop_event = threading.Event()
    failure = []

    worker = threading.Thread(
        target=_writer_entrypoint,
        args=(args, ready_event, stop_event, failure),
        daemon=True,
    )
    worker.start()

    if not ready_event.wait(timeout=5):
        stop_event.set()
        raise RuntimeError('Timed out waiting for initial samples.csv contents')
    if failure:
        raise failure[0]

    app = create_app(args.log_dir, int(max(args.refresh_seconds, 0.1) * 1000))
    try:
        run_app(
            app,
            args.host,
            args.port,
            open_browser=not args.no_open_browser,
            verbose=args.verbose,
        )
    finally:
        stop_event.set()
        worker.join(timeout=2)

    if failure:
        raise failure[0]


if __name__ == '__main__':
    main()
