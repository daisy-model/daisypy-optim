"""Aggregate functions.
Signature is (objectives : [float]) -> float
"""

def mean(objectives):
    """Mean value of objectives"""
    return sum(objectives) / float(len(objectives))

available_aggregate_fns = {
    "sum" : sum,
    "mean" : mean
}

