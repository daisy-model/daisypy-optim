'''Helpers for logging target time series used by objectives.'''
from daisypy.optim.scalar_objective import ScalarObjective
from daisypy.optim.multi_objective import MultiObjective

def log_targets(logger, objective_fn):
    '''Log all target time series exposed by an objective tree.'''
    for objective_name, outcome_name, target in _collect_targets(objective_fn):
        for row in target.dropna().itertuples(index=False):
            logger.target(
                objective_name=objective_name,
                outcome_name=outcome_name,
                time=row.time.isoformat(),
                target_value=row.value,
            )


def _collect_targets(objective_fn):
    if isinstance(objective_fn, MultiObjective):
        targets = []
        for child in objective_fn.objectives:
            targets.extend(_collect_targets(child))
        return targets
    if isinstance(objective_fn, ScalarObjective):
        return [(objective_fn.name, objective_fn.outcome_name, objective_fn.target)]
    return []
