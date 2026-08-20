from dataclasses import dataclass
from pathlib import Path

@dataclass
class StaticData:
    """Container for static simulation data

    Attributes
    ----------
    src : Path
      Absolute path to a file or a directory.

    dst : Path
      Relative path to a destination directory.
    """
    src : Path
    dst : Path

    def __init__(self, src, dst):
        """
        Parameters
        ----------
        src : str OR Path

        dst : str OR Path
        """
        self.src = Path(src).resolve()
        self.dst = Path(dst)
        assert not self.dst.is_absolute(), "dst_dir MUST be a relative path"
