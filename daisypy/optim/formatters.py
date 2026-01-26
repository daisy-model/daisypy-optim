'''Functions that formats python objects as string'''

def quote_if_string(value):
    '''Format a value as a string. Add quotes if value is already a string'''
    if isinstance(value, str):
        return '"' + value + '"'
    return str(value)
