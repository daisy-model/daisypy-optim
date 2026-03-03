from ax.api.configs import RangeParameterConfig, ChoiceParameterConfig
from .parameter import ContinuousParameter, CategoricalParameter

def daisy_param_to_ax_param(param):
    '''Convert Daisy parameter to Ax parameter

    Parameters
    ----------
    param : ContinuousParameter OR CategoricalParameter

    Returns
    -------
    RangeParameterConfig OR ChoiceParameterConfig
    '''
    if isinstance(param, ContinuousParameter):
        return RangeParameterConfig(
            name=param.name, parameter_type='float', bounds=param.valid_range
        )
    if isinstance(param, CategoricalParameter):
        init_value = param.values[param.initial_valud_idx]
        if isinstance(init_value, int):
            p_type = 'int'
        elif isinstance(init_value, float):
            p_type = 'float'
        elif isinstance(init_value, str):
            p_type = 'str'
        elif isinstance(init_value, bool):
            p_type = 'bool'
        else:
            raise ValueError('param values must be of type int, float, str or bool')
        return ChoiceParameterConfig(
            name=param.name, parameter_type=p_type, values=param.values, is_ordered=False)
    raise ValueError('param must of type ContinuousParameter or CategoricalParameter')
