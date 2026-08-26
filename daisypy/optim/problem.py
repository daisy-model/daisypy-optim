import tempfile
import os
import platform
from pathlib import Path
from daisypy.optim.output_store import OutputStore

class DaisyOptimizationProblem:
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-few-public-methods,too-many-instance-attributes
    '''A DaisyOptimizationProblem maps parameters to objectives, and is defined by
      - a list of simulations that should be "run as one"
      - a runner that knows how to run Daisy
      - a set of parameters
      - an objective
    '''
    def __init__(self,
                 runner,
                 simulations,
                 outcome_specs,
                 post_processing,
                 objective_fn,
                 parameters,
                 data_dir=None,
                 debug=False):
        """
        Parameters
        ----------
        runner : DaisyRunner

        simulations : dict of (str, Simulation)
          A dict of named simulations to run using the same set of parameters

        outcome_specs : {str : (str, str, str)}
          A dict defining with named outcomes. Each outcome is defined by a triplet of strings,
          (simulation, outcome, variable) that uniquely identifies a column in a outcome file

        objective_fn : daisypy.optim.Objective
          An objective function that computes one or more named objective values from a dict of
          named DataFrames.

        parameters : [DaisyParameter] OR dict of (str, [DaisyParameter])
          Parameters to optimize. If a list it is assumed that all parameters are for the 'dai' file

        data_dir : str
          If not None then temporary directories will be created in this directory. Otherwise, they
          will be created in a default location depending on platform.

        debug: bool
          If True do not delete the temporary directory where Daisy output is stored
        """
        self.runner = runner
        self.simulations = simulations
        self.outcome_specs = outcome_specs
        self.post_processing = post_processing
        self.objective_fn = objective_fn
        self.parameter_kind = {}
        if not isinstance(parameters, dict):
            parameters = { 'dai' : parameters }

        # Convert dict of parameters to a list of parameters and verify that there are no name
        # clashes.
        self.parameters = []
        for kind, params in parameters.items():
            for param in params:
                if param.name in self.parameter_kind:
                    raise ValueError(
                        'Parameters must have unique names across all templates. '
                        f'{param.name} from {kind} is also in {self.parameter_kind[param.name]}'
                    )
                self.parameter_kind[param.name] = kind
                self.parameters.append(param)

        self.data_dir = data_dir
        if data_dir is None and platform.system().lower() == 'linux':
            # There is a good chance that we are using flatpak, in which case we need to use a tmp
            # location that flatpak Daisy can read and write. Anything inside the user home is good.
            self.data_dir = os.path.expanduser('~/.tmp/daisy')
        if self.data_dir is not None:
            os.makedirs(self.data_dir, exist_ok=True)
        self.debug = debug

    def __call__(self, parameter_values):
        # TODO: Rewrite to accept a dict of parameters. This is too brittle
        """Run Daisy with the given parameters and evaluate the objective.

        Parameters
        ----------
        parameter_values : sequence
          Parameter values. Length MUST match length of ``self.parameters``.

        Returns
        -------
        ({ str : float }, { str : pandas.DataFrame }, { str : CompletedProcess })
          Triple of dicts, the first dict holds named objective values, the second dict holds named
          outcomes, the third dicts holds errors for each simulation
        """
        named_parameters = { 'runfile' : {} }
        for p, value in zip(self.parameters, parameter_values):
            kind = self.parameter_kind[p.name]
            if kind not in named_parameters:
                named_parameters[kind] = { p.name : value }
            else:
                named_parameters[kind][p.name] = value

        # If we debug then we dont want the directory to be deleted after use
        with tempfile.TemporaryDirectory(dir=self.data_dir, delete=not self.debug) as sim_dir:
            return self._run(sim_dir, named_parameters)

    def _run(self, base_sim_dir, named_parameters):
        '''Run Daisy in ``base_sim_dir`` and evaluate the objective on the produced files.'''
        base_sim_dir = Path(base_sim_dir)
        # We need to run all simulations in the problem before we can evaluate the objective

        errors = {}
        for sim_name, sim in self.simulations.items():
            sim_dir = base_sim_dir / sim_name
            # This will fail if there already is a dir with `sim_name` in the base dir, something
            # that should not be possible so we want a failure if it happens
            sim_dir.mkdir()
            sim_file = sim.setup(sim_dir, named_parameters)
            sim_result = self.runner(sim_file, sim_dir)
            if sim_result.returncode != 0:
                errors[sim_name] = sim_result
        output_store = OutputStore(self.simulations)
        outcomes = {
            name : output_store.extract(*spec) for name, spec in self.outcome_specs.items()
        }
        for name, p in self.post_processing.items():
            outcomes[name] = p(outcomes)
        return self.objective_fn(outcomes), outcomes, errors
