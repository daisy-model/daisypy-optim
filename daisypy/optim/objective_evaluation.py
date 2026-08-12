from dataclasses import dataclass, field


@dataclass
class ObjectiveEvaluation:
    '''Structured objective evaluation result.

    Attributes
    ----------
    objectives : dict[str, float]
      Named scalar objective values.
    predictions : dict[str, object]
      Exact extracted predictions keyed by objective name. Values are typically pandas.DataFrame
      objects with columns ``time`` and ``value``.
    '''
    objectives: dict[str, float]
    predictions: dict[str, object] = field(default_factory=dict)
