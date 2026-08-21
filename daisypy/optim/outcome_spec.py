from dataclasses import dataclass

@dataclass
class OutcomeSpec:
    """Specfication of an outcome in terms of a path to a location in in an OutputStore

    Attributes
    ----------
    sim : str
      Name of simulation

    output : str
      Name of output

    var : str
      Name of variable
    """
    sim : str
    output : str
    var : str
