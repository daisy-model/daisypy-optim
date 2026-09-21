from pathlib import Path
from daisypy.optim.file_generator import FileGenerator
from daisypy.optim.util import StrictFormatter

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
        self._formatter = StrictFormatter()
        self.out_file = out_file

        # Validate and set sub_dir, will throw if not a relative path
        self.sub_dir(Path("." if sub_dir is None else sub_dir))

        if template_file_path is not None:
            with open(template_file_path, 'r', encoding='utf-8') as infile:
                # Skip python line comments
                self.template_text = ''.join((
                    line for line in infile if not line.lstrip().startswith('#')
                ))
        else:
            self.template_text = template_text

    def __call__(self, output_directory, params):
        """Generate a python file from the template using the given params and write it to a
        directory

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
        py_string = self._formatter.format(self.template_text, **params)
        with open(out_path, "w", encoding='utf-8') as f:
            f.write(py_string)
        return out_path

    def relative_out_path(self):
        """Return the relative path the generated py files will be written to"""
        return self._sub_dir / self.out_file

    def sub_dir(self, path=None):
        """Get/set the sub dir"""
        if path is not None:
            path = Path(path)
            try:
                self._sub_dir = path.resolve().relative_to(Path.cwd(), walk_up=False)
            except ValueError as e:
                raise ValueError(f'"{path}" is not relative') from e
        return self._sub_dir
