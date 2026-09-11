from pathlib import Path

class OutputSpec:
    """Container for keeping track of simulation output paths.

    The point of this class is to keep things simple for users while allowing for reorganization of
    the simulation file hierarchy and subsequent rebasing when the simulation is actually running .

    Attributes
    ----------
    log : Path
      Path to the log file relative to the simulation root directory.

    var : str or [str]
      Name of variable or variables in log file
    """
    log : Path
    var : [str]

    def __init__(self, log, var, root=None):
        """In general, users of this class need not specify root, it is updated internally to
        handle rebasing of the file hierarchy

        Parameters
        ----------
        log : str OR Path

        var : str OR [str]

        root : str OR Path OR None
          Absolute path to root
        """
        self.log = Path(log)
        self.var = var if isinstance(var, list) else [var]
        self._root = Path("." if root is None else root).resolve()
        if self.log.is_absolute():
            raise ValueError("log MUST be a relative path")

    def path(self):
        """Get the absolute path to the log file

        Returns
        -------
        Path
        """
        return self._root / self.log

    def __repr__(self):
        return repr((self.path(), self.var))
