from .file_generator import FileGenerator

class MultiFileGenerator(FileGenerator):
    def __init__(self, generators):
        """Wrapper that handles multiple file generators

        Parameters
        ----------
        generators : dict of (str, FileGenerator)
        """
        self.generators = generators

    def __call__(self, output_directory, params):
        """Generate files

        Parameters
        ----------
        output_directory : str
          Directory to store the generated files in

        params : dict of (str, dict)
          Dictionary of parameters. Keys MUST match generator names.
          Keys in nested dictionaries MUST match the corresponding generators parameters

        Returns
        -------
        paths : dict of (str, str)
          Map from generator name to generated file
        """
        paths = {}
        for gen_name, gen_params in params.items():
            paths[gen_name] = self.generators[gen_name](output_directory, gen_params, False)
        return paths
