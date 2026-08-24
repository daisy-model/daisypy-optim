'''Helpers for logging extracted model outcomes.'''


def log_outcomes(logger, outcomes, **context):
    '''Log outcome rows

    Parameters
    ----------
    logger : Logger
      Logger used to write outcome rows.
    outcomes : { str : pd.DataFrame }
      Named outcomes. Each outcome MUST have columns "time" and "value"
    **context
      Extra columns to include in every logged row, for example ``evaluation_id`` or ``step``.
    '''
    for outcome_name, outcome in outcomes.items():
        for row in outcome.itertuples(index=False):
            logger.outcome(
                **context,
                outcome_name=outcome_name,
                time=row.time.isoformat(),
                predicted_value=row.value,
            )
