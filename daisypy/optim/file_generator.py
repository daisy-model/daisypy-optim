import os

class DaiFileGenerator:
    def __init__(self, template_text='', template_file_path=None, out_file='runfile.dai'):
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
        template_text : str
          Template text.

        template_file_path : str
          Path to template. Overrides template_text if no None

        out_file : str
          Name to use for generated file
        """
        self.out_file = out_file
        if template_file_path is not None:
            with open(template_file_path, 'r', encoding='utf-8') as infile:
                # Skip dai line comments
                self.template_text = ''.join((
                    line for line in infile if not line.lstrip().startswith(';')
                ))
        else:
            self.template_text = template_text

    def __call__(self, output_directory, params):
        """Generate a dai file from the template using the given params and write it to a directory

        Parameters
        ----------
        output_directory : str
          Directory to store the generated file in

        params : dict
          Dictionary of parameters. Keys MUST match the defined template parameters exactly

        Returns
        -------
        out_path : Path to the generated file
        """
        os.makedirs(output_directory, exist_ok=True)
        dai_string = self.template_text.format(**params)
        out_path = os.path.join(output_directory, self.out_file)
        with open(out_path, "w", encoding='utf-8') as f:
            f.write(dai_string)
        return os.path.abspath(out_path)

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
