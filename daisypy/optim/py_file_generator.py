import os
from pathlib import Path
from .file_generator import FileGenerator

class PyFileGenerator(FileGenerator):
    """Template based generation of python files using string replacement

    Parameters in the template are specifed in curly braces {}. For example,

        active_depth = {active_soil_layer_depth}

    Which specifies a parameter called `active_soil_layer_depth`.

    Note:
    In the template, you need to use double braces in sets and f-strings.

    my_set = {{ my_var }}
    my_string = f'{{ my_var }}
    """
    def __init__(self, out_file, template_text='', template_file_path=None, sub_dir=None):
        """
        Parameters
        ----------
        out_file : str
          Name to use for generated file. This needs to match what you use in the dai file

        template_text : str
          Template text.

        template_file_path : str
          Path to template. Overrides template_text if not None

        sub_dir : str or None
          If not None generate files in this subdirectory otherwise generate in root of outdir
        """
        self.out_file = out_file
        self.sub_dir = Path("." if sub_dir is None else sub_dir)
        # Verify that it is an actual sub dir. Will throw ValueError if not
        self.sub_dir.resolve().relative_to(Path.cwd(), walk_up=False)
        if template_file_path is not None:
            with open(template_file_path, 'r', encoding='utf-8') as infile:
                # Skip python line comments
                self.template_text = ''.join((
                    line for line in infile if not line.lstrip().startswith('#')
                ))
        else:
            self.template_text = template_text

    def __call__(self, output_directory, params, tagged=True):
        """Generate a python file from the template using the given params and write it to a
        directory

        Parameters
        ----------
        output_directory : str
          Directory to store the generated file in

        params : dict (str, value) OR { 'py' : dict (str, value) }
          If tagged is True, then the key 'py' MUST be in params and the value MUST be a dict of
          parameters, where the keys MUST match the defined template parameters exactly.
          If tagged is False, then the keys MUST match the defined template parameters exactly.

        tagged : bool
          If True return a tagged path otherwise return a plain path

        Returns
        -------
        { 'py' : out_path } OR out_path
        """
        if tagged:
            params = params['py']
        output_directory = (Path(output_directory) / self.sub_dir).resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        out_path = output_directory / self.out_file
        py_string = self.template_text.format(**params)
        with open(out_path, "w", encoding='utf-8') as f:
            f.write(py_string)
        if tagged:
            return { 'py' : out_path }
        return out_path

    def relative_out_path(self):
        """Return thee relative path the generated py files will be written to"""
        return os.path.join(self.sub_dir, self.out_file)

    def copy_and_update(self, **kwargs):
        return PyFileGenerator(
            kwargs.get("out_file", self.out_file),
            kwargs.get("template_text", self.template_text),
            kwargs.get("template_file_path", None),
            kwargs.get("sub_dir", self.sub_dir)
        )

    def serialize(self):
        '''Serializable representation of this PyFileGenerator

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
        '''Create a PyFileGenerator from a serialized representation

        Parameters
        ----------
        dict_repr: dict of (str, str)
          dict with keys 'template_text' and 'out_file'
        '''
        return PyFileGenerator(template_text=dict_repr['template_text'],
                                out_file=dict_repr['out_file'])
