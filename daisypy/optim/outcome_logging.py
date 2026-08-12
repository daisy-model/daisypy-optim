'''Helpers for logging extracted model predictions.'''


def log_outcomes(logger, evaluation, **context):
    '''Log the extracted prediction rows from an objective evaluation.

    Parameters
    ----------
    logger : Logger
      Logger used to write outcome rows.
    evaluation : ObjectiveEvaluation
      Structured objective evaluation containing extracted predictions.
    **context
      Extra columns to include in every logged row, for example ``evaluation_id`` or ``step``.
    '''
    for objective_name, prediction in evaluation.predictions.items():
        for row in prediction.itertuples(index=False):
            logger.outcome(
                **context,
                objective_name=objective_name,
                time=row.time.isoformat(),
                predicted_value=row.value,
            )
