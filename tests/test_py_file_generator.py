# pylint: disable=missing-function-docstring
from pathlib import Path
from daisypy.optim.py_file_generator import PyFileGenerator

EXPECTED = """def linear(x):
    return 0.5 * x + 10
"""
PARAMS = {'a' : 0.5, 'b' : 10}

def test_tagged(tmp_path):
    template_path = Path(__file__).parent / 'templates' / 'template.py'
    generator = PyFileGenerator('testing.py', template_file_path=template_path)
    file_path = Path(generator(tmp_path, {'py' : PARAMS})['py'])
    assert file_path.read_text(encoding='utf-8') == EXPECTED

def test_not_tagged(tmp_path):
    template_path = Path(__file__).parent / 'templates' / 'template.py'
    generator = PyFileGenerator('testing.py', template_file_path=template_path)
    file_path = Path(generator(tmp_path, PARAMS, tagged=False))
    assert file_path.read_text(encoding='utf-8') == EXPECTED

def test_no_params(tmp_path):
    template = "x = {{ 'a' : 1 }}"
    expected = "x = { 'a' : 1 }"
    generator = PyFileGenerator('testing.py', template_text=template)
    file_path = Path(generator(tmp_path, {}, tagged=False))
    assert file_path.read_text(encoding='utf-8') == expected
    file_path = Path(generator(tmp_path, {'py' : {}})['py'])
    assert file_path.read_text(encoding='utf-8') == expected
