import pandas as pd

class DaisyLoss:
    def __init__(self, loss_fn):
        """Loss wrappeer for use with DaisyObjective

        Parameters
        ----------
        loss_fn : callable : (actual, target) -> loss
          actual and target are numpy.ndarray of shape (n,)
        """
        self.loss_fn = loss_fn

    def __call__(self, actual, target):
        """Compute the loss over a timeseries

        Timestamps MUST be unique in both actual and target.
        All timestamps in target MUST be in actual.
        Extra timestamps in actual are ignored.

        Parameters
        ----------
        actual : pd.DataFrame
          Must contain columns "time" and "value"

        target : pd.DataFrame
          Must contain columns "time" and "value"

        Raises
        ------
        ValueError if timestamps are not unique or if actual is missing timestamps

        Returns
        -------
        loss as computed by self.loss_fn
        """
        # Only compute loss for time point with measurements
        target = target.dropna()
        target_time = target['time']
        if target_time.nunique() != len(target_time):
            raise ValueError('Timestamps in target must be unique')

        actual_time = actual['time']
        if actual_time.nunique() != len(actual_time):
            raise ValueError('Timestamps in actual must be unique')

        if not target_time.isin(actual_time).all():
            raise ValueError('All timestamps in target must be in actual')

        merged = pd.merge(
            target, actual,
            how='left',
            on='time',
            suffixes=('_target', '_actual')
        )
        return self.loss_fn(merged['value_actual'].to_numpy(), merged['value_target'].to_numpy())
