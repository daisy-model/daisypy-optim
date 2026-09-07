import sys
import argparse
from pathlib import Path
try:
    from daisypy.optim.monitor import create_app, run_app
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
        try:
            run_app(
                app,
                args.host,
                args.port,
                open_browser=not args.no_open_browser,
                verbose=args.verbose,
            )
            return 0
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f'Error running monitor: {e}', file=sys.stderr)
            return 1

except ModuleNotFoundError as e:
    ERR_MSG = f'Error: {e}'
    def main():
        '''Print an error message about missing dependencies and how to install them'''
        print(
            ERR_MSG,
            'A likely cause of this error is that dependencies for the `monitor` program are not '
            'installed by default.',
            'To install these dependencies run:',
            '    pip install "daisypy-optim[monitor]"',
            sep='\n', file=sys.stderr
        )
        return 2

if __name__ == '__main__':
    sys.exit(main())
