import os
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
    def __init__(self, out_file, template_text='', template_file_path=None):
        """
        Parameters
        ----------
        out_file : str
          Name to use for generated file. This needs to match what you use in the dai file

        template_text : str
          Template text.

        template_file_path : str
          Path to template. Overrides template_text if not None
        """
        self.out_file = out_file
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
        os.makedirs(output_directory, exist_ok=True)
        py_string = self.template_text.format(**params)
        out_path = os.path.abspath(os.path.join(output_directory, self.out_file))
        with open(out_path, "w", encoding='utf-8') as f:
            f.write(py_string)
        if tagged:
            return { 'py' : out_path }
        return out_path

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
