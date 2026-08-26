import pandas as pd
from daisypy.optim.post_processor import PostProcessor
from daisypy.optim.util import check_outcomes, merge_outcomes

__all__ = [
    "AggregateOutcomes"
]

class AggregateOutcomes(PostProcessor):
    # pylint: disable=too-few-public-methods
    """Aggregate outcomes"""

    def __init__(self, outcome_names, aggregate_fn):
        """
        Parameters
        ----------
        outcome_names : [str]
          List of outcomes to aggregate

        aggregate_fn : Callable [[pandas.Series], float]
          Aggregation function. It is passed a series with named values, where the names correspond
          to the outcome names,
        """
        self.outcome_names = outcome_names
        self.aggregate_fn = aggregate_fn

    def __call__(self, outcomes):
        """Aggregate the outcomes
        Parameters
        ----------
        outcomes : { str : pandas.DataFrame }
          Named outcome DataFrames

        Returns
        -------
        pandas.DataFrame with columns "time" and "value"
        """
        try:
            outcomes_subset = { k : outcomes[k] for k in self.outcome_names }
        except KeyError as e:
            raise ValueError(f"`outcomes` does not contain '{e.args[0]}'") from e
        check_outcomes(outcomes_subset)
        merged = merge_outcomes(outcomes_subset)
        return pd.DataFrame({
            "time" : merged["time"],
            "value" : merged.drop(columns=["time"]).aggregate(self.aggregate_fn, axis="columns")
        })
