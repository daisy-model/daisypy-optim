'''Example illustrating how to define a simulation and manually generate simulation files, run the
simulation and inspect outputs.
The simulation uses the following file hierarchy
./
  setup.py
  common/
    weather.dwf
  sim/
    example.dai

The example.dai file contains a path specification in the top
(path "./common/" "../common"  &old)

further down it contains a weather specification
(defweather Normal table (file "weather.dwf"))

Without the path specification, Daisy will look for "weather.dwf" in the directory it runs from.
With the path specification it will look for `common/weather.dwf` and `../common/weather.dwf`.
The reason it needs the `../common` part as well is because it defines a spawn program that is run
from auto generated sub directories.

It is easy to make mistakes with these paths. So a good idea is to run a script like this to verify
that the simulation is defined correctly before starting an optimization.
'''
import argparse
from pathlib import Path
from daisypy.optim import (
    DaiFileGenerator,
    DaisyRunner,
    OutputSpec,
    Simulation,
    StaticData,
)
from daisypy.optim.output_store import OutputStore

def setup(base_dir):
    '''Setup the Simulation'''
    # We will generate a single dai file in a subdirectory called sim
    file_generators = {
        "runfile" : DaiFileGenerator(
            "run.dai",
            template_file_path=base_dir / 'sim' / 'example.dai',
            sub_dir='sim'
        ),
    }

    # We want to copy the entire common dir
    static_data = [
        StaticData(base_dir / "common", "./"),
    ]

    # The first argument to OutputSpec is path to a dlf file relative to the simulation root dir
    # The location of dlf files is controlled by whether or not we are using spawn and by the
    # log_prefix set in the dai file.
    # The second argument to OutputSpec is either a single variable name or a list of variable
    # names. These names must be in the file.
    outputs = {
        'Askov/water' : OutputSpec('Askov/field_water.dlf', 'Surface water'),
        'Jyndevad/water' : OutputSpec('Jyndevad/field_water.dlf', 'Surface water'),
    }

    return Simulation(file_generators, outputs, static_data=static_data)

def run(base_dir, sim, daisy_path):
    '''Run the simulation'''
    # We can test the simulation setup by generating the files and then running daisy
    run_dir = base_dir / 'tmp'
    dai_file = sim.setup(run_dir, {"runfile": {}})

    # You need to substitute the path to your Daisy executable
    runner = DaisyRunner(daisy_path)
    result = runner(dai_file, run_dir)
    if result.returncode == 0:
        print('Simulation completed succesfully')
        output_store = OutputStore({ 'sim' : sim })
        print('Stored these outputs')
        for sim_name, sim_outputs in output_store.items():
            print(f'# {sim_name} #')
            for output_name, output_value in sim_outputs.items():
                print(f'## {output_name} ##')
                print(output_value)
                print('-' * 80)
    else:
        print('Simulation failed')
        print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('daisy', nargs='?', type=str, help='Path to Daisy', default='daisy')
    args = parser.parse_args()
    sim_dir = Path(__file__).parent
    simulation = setup(sim_dir)
    run(sim_dir, simulation, args.daisy)
