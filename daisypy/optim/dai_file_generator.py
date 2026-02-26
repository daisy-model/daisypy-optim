import os
import warnings
from pathlib import Path
from .file_generator import FileGenerator
from daisypy.io import parse_dai, format_dai, filter_dai
from daisypy.io.dai import Definition, Comment
from daisypy.io.exceptions import DaiException

class DaiFileGenerator(FileGenerator):
    def __init__(self, out_file='run.dai', template_text='', template_file_path=None):
        """Template based generation of dai files using string replacement

        Parameters in the template are specifed in curly braces {}. For example,

          ...
          (Groundwater aquitard
            (K_aquitard {K_aquitard_param} [mm/d])
            ...
          )

        Which specifies a parameter called `K_aquitard_param`


        Parameters
        ----------
        out_file : str
          Name to use for generated file

        template_text : str
          Template text.

        template_file_path : str
          Path to template. Overrides template_text if no None

        tag : str
          Tag to use when returning generated paths
        """
        self.out_file = out_file
        if template_file_path is not None:
            template_text = Path(template_file_path).read_text()
        # Parse the text as a Dai object while allowing placeholders
        dai = parse_dai(template_text, extended=True)
        dai = filter_dai(dai, lambda x : not isinstance(x, Comment))

        # Force all programs that inherits from spawn to run with 1 process
        for value in dai.values:
            if isinstance(value, Definition) and value.parent.value == 'spawn':
                for param in value.body:
                    if isinstance(param, list) and param[0].value == 'parallel':
                        if param[1] != 1:
                            warnings.warn("parallel parameter for spawn forced to 1")
                            param[1] = 1
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
        os.makedirs(output_directory, exist_ok=True)
        dai_string = self.template_text.format(**params)
        out_path = os.path.abspath(os.path.join(output_directory, self.out_file))
        with open(out_path, "w", encoding='utf-8') as f:
            f.write(dai_string)
        if tagged:
            return { 'dai' : out_path }
        return out_path

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
