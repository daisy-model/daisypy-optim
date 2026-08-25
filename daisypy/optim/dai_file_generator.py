# pylint: disable=R0801
import os
import warnings
from pathlib import Path
from daisypy.io import parse_dai, format_dai, filter_dai
from daisypy.io.dai import Definition, Comment, Identifier
from daisypy.optim.file_generator import FileGenerator
from daisypy.optim.util import StrictFormatter

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
        self.sub_dir = Path("." if sub_dir is None else sub_dir)
        # Verify that it is an actual sub dir. Will throw ValueError if not
        self.sub_dir.resolve().relative_to(Path.cwd(), walk_up=False)
        if template_file_path is not None:
            template_text = Path(template_file_path).read_text(encoding='utf-8')
        # Parse the text as a Dai object while allowing placeholders
        dai = parse_dai(template_text, extended=True)
        dai = filter_dai(dai, lambda x : not isinstance(x, Comment))

        # Force all programs that inherits from spawn to run with 1 process
        for value in dai.values:
            if isinstance(value, Definition) and value.parent.value == 'spawn':
                has_parallel = False
                for param in value.body:
                    if isinstance(param, list) and param[0].value == 'parallel':
                        has_parallel = True
                        if param[1] != 1:
                            warnings.warn("parallel parameter for spawn forced to 1")
                            param[1] = 1
                if not has_parallel:
                    warnings.warn("parallel parameter for spawn forced to 1")
                    value.body.append([Identifier('parallel'), 1])
        self.template_text = format_dai(dai)

    def __call__(self, output_directory, params, tagged=True):
        """Generate a dai file from the template using the given params and write it to a directory

        Parameters
        ----------
        output_directory : str
          Directory to store the generated file in

        params : dict (str, value) OR { 'dai' : dict (str, value) }
          If tagged is True, then the key 'dai' MUST be in params and the value MUST be a dict of
          parameters, where the keys MUST match the defined template parameters exactly.
          If tagged is False, then the keys MUST match the defined template parameters exactly.

        tagged : bool
          If True return a tagged path otherwise return a plain path

        Returns
        -------
        { 'dai' : out_path } OR out_path
        """
        if tagged:
            params = params['dai']
        output_directory = (Path(output_directory) / self.sub_dir).resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        out_path = output_directory / self.out_file
        dai_string = self._formatter.format(self.template_text, **params)
        with open(out_path, "w", encoding='utf-8') as f:
            f.write(dai_string)
        if tagged:
            return { 'dai' : out_path }
        return out_path

    def relative_out_path(self):
        """Return the relative path the generated dai files will be written to"""
        return os.path.join(self.sub_dir, self.out_file)

    def copy_and_update(self, **kwargs):
        return DaiFileGenerator(
            kwargs.get("out_file", self.out_file),
            kwargs.get("template_text", self.template_text),
            kwargs.get("template_file_path", None),
            kwargs.get("sub_dir", self.sub_dir)
        )

    def serialize(self):
        '''Serializable representation of this DaiFileGenerator

        Returns
        -------
        dict of (str, str)
        '''
        return {
            'template_text' : self.template_text,
            'out_file' : self.out_file
        }

    @staticmethod
    def unzerialize(dict_repr):
        '''Create a DaiFileGenerator from a serialized representation

        Parameters
        ----------
        dict_repr: dict of (str, str)
          dict with keys 'template_text' and 'out_file'
        '''
        return DaiFileGenerator(template_text=dict_repr['template_text'],
                                out_file=dict_repr['out_file'])
