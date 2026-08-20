from pathlib import Path

class OutputSpec:
    """Container for keeping track of simulation output paths.

    The point of this class is to keep things simple for users while allowing for reorganization of
    the simulation file hierarchy and subsequent rebasing when the simulation is actually running .

    Attributes
    ----------
    log : Path
      Path to the log file relative to the simulation file that produces the log.

    var : str or [str]
      Name of variable or variables in log file
    """
    log : Path
    var : [str]

    def __init__(self, log, var, sub_dir=None, root=None):
        """In general, users of this class need not specify sub_dir and root, these are used by the
        library internally to handle rebasing of the file hierarchy

        Parameters
        ----------
        log : str OR Path

        var : str OR [str]

        sub_dir : str OR Path OR None
          Relative path stub that relates the log path to the root

        root : str OR Path OR None
          Absolute path to root
        """
        self.log = Path(log)
        self.var = var if isinstance(var, list) else [var]
        self._sub_dir = Path("." if sub_dir is None else sub_dir)
        self._root = Path("/" if root is None else root)
        assert not self.log.is_absolute(), "log MUST be a relative path"
        assert not self._sub_dir.is_absolute(), "sub_dir MUST be a relative path"
        # Verify that the sub_dir path is not pointing to a directory above, e.g. a/../../"
        # In that case Path. relative_to will throw a ValueError when walk_up=False
        self._sub_dir.resolve().relative_to(Path.cwd(), walk_up=False)
        assert self._root.is_absolute(), "root MUST be absolute"

    def path(self):
        """Get the absolute path to the log file

        Returns
        -------
        Path
        """
        return self._root / self._sub_dir / self.log

    def __repr__(self):
        return repr((self.path(), self.var))
