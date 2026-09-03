'''Manual interactive test for live sample plotting.'''
import argparse
import sys
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from daisypy.optim.csv_log import CsvLog
from daisypy.optim.plot_samples import run as run_plot_samples
# pylint: enable=wrong-import-position


def _make_standardized(values, centre, scale):
    return (values - centre) / scale


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


def _write_live_samples(
        log_dir,
        write_interval,
        ready_event,
        stop_event,
        include_standardized,
):
    # pylint: disable=too-many-arguments, too-many-locals
    samples_path = Path(log_dir) / 'samples.csv'
    columns = [
        'step',
        'index',
        'tag',
        'metric_loss',
        'param_x',
        'param_y',
    ]
    rng = np.random.default_rng()
    optimum = np.array([2.0, -0.8], dtype=float)
    centre = optimum + rng.normal(loc=[3.5, 1.2], scale=[1.0, 0.5])
    scale = np.array([1.8, 0.7], dtype=float)

    with CsvLog(samples_path, columns=columns) as log:
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
                log.log(
                    step=step,
                    index=index,
                    tag='raw',
                    metric_loss=loss,
                    param_x=sample[0],
                    param_y=sample[1],
                )
                if include_standardized:
                    log.log(
                        step=step,
                        index=index,
                        tag='standardized',
                        metric_loss=loss,
                        param_x=sample_standardized[0],
                        param_y=sample_standardized[1],
                    )

            if step == 0:
                ready_event.set()
            step += 1
            if stop_event.wait(write_interval):
                break


def _stop_after_delay(delay, stop_event):
    time.sleep(delay)
    stop_event.set()
    plt.close('all')


def main():
    '''Run a manual live-plotting scenario for plot_samples.'''
    parser = argparse.ArgumentParser(
        description='Manual interactive test that continuously updates samples.csv for plot_samples'
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path('manual-tests/out/plot-samples'),
        help='Directory where samples.csv is written.',
    )
    parser.add_argument(
        '--poll-interval',
        type=float,
        default=1,
        help='Seconds between plot refresh checks.',
    )
    parser.add_argument(
        '--write-interval',
        type=float,
        default=2,
        help='Seconds between new optimization-like sample batches.',
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Optional runtime in seconds before the script stops automatically.',
    )
    parser.add_argument(
        '--standardized',
        action='store_true',
        default=False,
        help='Plot standardized parameter values and include standardized rows in samples.csv.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Optional output image path forwarded to plot_samples.',
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

    if args.duration is not None:
        threading.Thread(
            target=_stop_after_delay,
            args=(args.duration, stop_event),
            daemon=True,
        ).start()

    print(f'Writing live samples to {args.log_dir / "samples.csv"}')
    print('Close the plot window or press Ctrl+C to stop.')
    try:
        run_plot_samples(
            args.log_dir,
            standardized=args.standardized,
            output_path=args.output,
            poll_interval=args.poll_interval,
        )
    finally:
        stop_event.set()
        plt.close('all')
        worker.join(timeout=2)

    if failure:
        raise failure[0]


def _writer_entrypoint(args, ready_event, stop_event, failure):
    try:
        _write_live_samples(
            args.log_dir,
            args.write_interval,
            ready_event,
            stop_event,
            args.standardized,
        )
    except Exception as exc: # pylint: disable=broad-exception-caught
        failure.append(exc)
        ready_event.set()


if __name__ == '__main__':
    main()
