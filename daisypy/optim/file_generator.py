import os

class DaiFileGenerator:
    def __init__(self, template_file_path, generated_file_name='runfile.dai'):
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
        template_file_path : str
          Path to template

        generated_file_name : str
          Name to use for generated file
        """
        self.out_file = generated_file_name
        with open(template_file_path, 'r', encoding='utf-8') as file:
            self.template_text = file.read()

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
