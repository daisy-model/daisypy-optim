'''Helpers for logging target time series used by objectives.'''


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
    if hasattr(objective_fn, 'objective_fns'):
        targets = []
        for child in objective_fn.objective_fns:
            targets.extend(_collect_targets(child))
        return targets
    if hasattr(objective_fn, 'target') and hasattr(objective_fn, 'name'):
        return [(objective_fn.name, objective_fn.outcome_name, objective_fn.target)]
    return []
