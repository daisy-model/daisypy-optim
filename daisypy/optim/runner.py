import os
import subprocess

class DaisyRunner:
    def __init__(self, daisy_bin, daisy_home):
        self.daisy_bin = daisy_bin
        self.daisy_home = daisy_home

    def __call__(self, dai_file, output_directory):
        args = [
            self.daisy_bin,
            "-d", output_directory,
            dai_file
        ]
        result = subprocess.run(args, env={"DAISYHOME" : self.daisy_home})
