import os
import time
import subprocess
from pathlib import Path

class DaisyRunner:
    # pylint: disable=too-few-public-methods
    """Class that knows how to run daisy"""

    def __init__(self, daisy_bin, daisy_home=None, max_tries=3, backoff=0.01):
        """
        Parameters
        ----------
        daisy_bin : str
          Path to daisy binary

        daisy_home : str
          Path to daisy home directory containing lib/ and sample/
          If not None set DAISYHOME environment variable to daisy_home. Otherwise dont set DAISYHOME


        """
        self.daisy_bin = daisy_bin
        if daisy_home is not None:
            os.environ.update('DAISYHOME', daisy_home)
        self.max_tries = max_tries
        self.backoff = backoff
        # The flatpak version of Daisy can fail due to a flatpak startup issue where the run is
        # aborted when openat2(...) returns -1 EAGAIN (Resource temporarily unavailable)
        # The problem is concurrency related, and can (often?/always?) be resolved by retrying.
        # The emitted error message from flatpak is
        # error: Extension org.freedesktop.Platform.GL.default has invalid merge-dirs
        self._retry_errors = set([
            b'invalid merge-dirs'
        ])

    def __call__(self, dai_file, output_directory=None):
        """Run daisy

        Parameters
        ----------
        dai_file : str
          Path to dai file to run

        output_directory : str or None
          Path to output directory, if None use the directory of the dai file as output directory

        Returns
        -------
        subprocess.CompletedProcess
        """
        if output_directory is None:
            output_directory = Path(dai_file).parent
        args = [
            self.daisy_bin,
            "-q",
            "-d", str(output_directory),
            str(dai_file)
        ]
        for i in range(max(1, self.max_tries)):
            if i > 0:
                print('Retrying simulation')
                time.sleep(self.backoff)
            result = subprocess.run(args, capture_output=True, check=False)
            if result.returncode == 0:
                break
            retry = False
            for retry_error in self._retry_errors:
                if result.stderr.find(retry_error) != -1:
                    # This is a known error so retry after a short wait.
                    print(f'Running "{dai_file}" failed ({i+1}/{self.max_tries})')
                    print(result.stderr)
                    retry = True
                    break # No need to check other strings, because we know we should retry
            if not retry:
                break
        return result
