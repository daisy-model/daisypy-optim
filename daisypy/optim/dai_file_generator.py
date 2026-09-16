# pylint: disable=R0801
from pathlib import Path
from daisypy.io import parse_dai, format_dai, filter_dai
from daisypy.io.dai import Definition, Comment, Identifier
from daisypy.optim.file_generator import FileGenerator
from daisypy.optim.util import StrictFormatter

SPAWN_PARALLEL_PARAM = '_spawn-parallel-param'

class DaiFileGenerator(FileGenerator):
    """Template based generation of dai files using string replacement

     Parameters in the template are specifed in curly braces {}. For example,

       ...
       (Groundwater aquitard
         (K_aquitard {K_aquitard_param} [mm/d])
         ...
       )

     Which specifies a parameter called `K_aquitard_param`
     """
    def __init__(self, out_file='run.dai', template_text='', template_file_path=None, sub_dir=None):
        """
        Parameters
        ----------
        out_file : str
          Name to use for generated file

        template_text : str
          Template text.

        template_file_path : str
          Path to template. Overrides template_text if not None

        sub_dir : str or None
          If not None generate files in this subdirectory otherwise generate in root of outdir
        """
        self._formatter = StrictFormatter()
        self.out_file = out_file

        # Validate and set sub_dir, will trow if not a relative path
        self.sub_dir(Path("." if sub_dir is None else sub_dir))

        if template_file_path is not None:
            self.template_text = Path(template_file_path).read_text(encoding='utf-8')
        else:
            self.template_text = template_text

        # Parse the text as a Dai object while allowing placeholders, then add
        # (parallel {_spawn-parallel-param}) to all spawn programs.
        self.has_spawn_program = False
        self.process_cost = 1
        dai = parse_dai(self.template_text, extended=True)
        dai = filter_dai(dai, lambda x : not isinstance(x, Comment))
        # Will update self.has_spawn_program if any are found
        dai = self._parameterize_spawn_programs(dai)
        self.template_text = format_dai(dai)

    def __call__(self, output_directory, params):
        """Generate a dai file from the template using the given params and write it to a directory

        Parameters
        ----------
        output_directory : str
          Directory to store the generated file in

        params : dict (str, value)
          A dict of parameters, where the keys MUST match the defined template parameters exactly.

        Returns
        -------
        out_path
        """
        output_directory = (Path(output_directory) / self._sub_dir).resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        out_path = output_directory / self.out_file
        dai_string = self._formatter.format(self.template_text, **params)
        with open(out_path, "w", encoding='utf-8') as f:
            f.write(dai_string)
        return out_path

    def sub_dir(self, path=None):
        if path is not None:
            path = Path(path)
            try:
                self._sub_dir = path.resolve().relative_to(Path.cwd(), walk_up=False)
            except ValueError as e:
                raise ValueError(f'"{path}" is not relative') from e
        return self._sub_dir

    def relative_out_path(self):
        """Return the relative path the generated dai files will be written to"""
        return self._sub_dir / self.out_file

    def _parameterize_spawn_programs(self, dai):
        n = 0
        for i, value in enumerate(dai.values):
            if _is_spawn(value):
                self.has_spawn_program = True
                n += _count_spawn_programs(value)
                dai.values[i] = _set_parallel(value, f'{{{SPAWN_PARALLEL_PARAM}}}')
        self.process_cost = max(1, n)
        return dai

def _is_spawn(dai):
    return (isinstance(dai, Definition) and
            dai.component.value == 'program' and
            dai.parent.value == 'spawn')

def _count_spawn_programs(spawn):
    for param in spawn.body:
        if _is_program(param):
            return len(param) - 1
    return 0

def _set_parallel(spawn, value):
    for param in spawn.body:
        if _is_parallel(param):
            param[1] = value
            return spawn
    spawn.body.append([Identifier('parallel'), value])
    return spawn

def _is_parallel(param):
    return isinstance(param, list) and len(param) == 2 and param[0].value == 'parallel'

def _is_program(param):
    return isinstance(param, list) and len(param) > 0 and param[0].value == 'program'
