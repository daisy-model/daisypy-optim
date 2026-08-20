import os
from pathlib import Path
from daisypy.optim.util import copy_into
from daisypy.optim.static_data import StaticData

__all__ = [
    'Simulation'
]

class Simulation:
    # pylint: disable=too-few-public-methods
    """Container holding everything required forrunning a simulation.
    The point of this class is to ensure that all
    files are in the expected locations for each simulation.

    In dai files it is common to include other files using relative paths, like
        (input file "../relative/path/to/other.dai")
    This does not work well when we create new temporary directories for each simulation. So we need
    to copy all the static files that are included, and ensure that they are in the right relative
    location.

    Relative paths that are at the parent ("../") or further up are handled by computing the full
    file tree and rooting it at the temporary directory. This means that the actual simulation
    file and the corresponding outputs are not necesarily in the base directory.
    """
    def __init__(self, file_generators, static_data=None):
        """
        Parameters
        ----------
        file_generators : dict of (str, FileGenerator)
          Named file generators

        static_data : [StaticData] or None
          Each static data path is copied to its relative destination directory.
          Paths can be files or directories. If a directory, then the entire directory is copied
          to the destination directory. If you wish to copy only the contents of the directory you
          will have to specify each file manually,
        """
        # TODO: We could have a situation where we only want to optimize parameters for a python
        # function. Then it would be nice to have an interface where we just copy the dai file as
        # static data.
        assert "dai" in file_generators, "There must be a generated dai file"
        self._generators = file_generators
        self._static_data = [] if static_data is None else static_data
        self._update_paths()

    def _update_paths(self):
        # Compute the path tree by assuming the current working dir is the root of all relative
        # paths
        if len(self._static_data) == 0:
            # Nothing to do when there is no static data
            return

        # We need both static paths and generated paths
        abs_paths = (
            [s.dst.resolve() for s in self._static_data] +
            [Path(g.relative_out_path()).resolve() for g in self._generators.values()]
        )
        root = os.path.commonpath(abs_paths)
        self.static_data = [
            StaticData(s.src, s.dst.resolve().relative_to(root)) for s in self._static_data
        ]

        generators = {}
        for g_name, g in self._generators.items():
            sub_dir = Path(g.relative_out_path()).resolve().relative_to(root).parent
            generators[g_name] = g.copy_and_update(sub_dir=sub_dir)
        self._generators = generators

    def __call__(self, output_directory, params):
        """Setup environment by copying static files and instantiating parameterized files

        Parameters
        ----------
        output_directory : str or Path
          Base directory to store the files in

        params : dict of (str, dict)
          Dictionary of parameters. Keys MUST match generator names.
          Keys in nested dictionaries MUST match the corresponding generators parameters

        Returns
        -------
        path to simulation file
        """
        assert params.keys() == self._generators.keys(), \
            "Keys in params must match generator names exactly"
        output_directory = Path(output_directory)
        for sd in self._static_data:
            dst = output_directory / sd.dst
            copy_into(sd.src, dst)
            # From python 3.14 we can use Path.copy_into
            # sd.src.copy_into(sd.dst, follow_symlinks=False)

        paths = {}
        for gen_name, gen_params in params.items():
            paths[gen_name] = self._generators[gen_name](output_directory, gen_params, False)

        return paths["dai"]
