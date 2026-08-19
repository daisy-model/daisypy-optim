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

def test_sub_dir(tmp_path):
    template_path = Path(__file__).parent / 'templates' / 'template.py'
    generator = PyFileGenerator('testing.py', template_file_path=template_path, sub_dir='nested/py')
    file_path = Path(generator(tmp_path, {'py' : PARAMS})['py'])
    assert file_path == tmp_path / 'nested' / 'py' / 'testing.py'
    assert file_path.read_text(encoding='utf-8') == EXPECTED

def test_relative_out_path():
    generator = PyFileGenerator('testing.py', template_text='', sub_dir='nested/py')
    assert generator.relative_out_path() == 'nested/py/testing.py'

def test_copy_and_update(tmp_path):
    generator = PyFileGenerator('testing.py', template_text='value = {value}', sub_dir='nested/py')
    copied = generator.copy_and_update(out_file='copied.py', sub_dir='updated')
    file_path = Path(copied(tmp_path, {'py' : {'value' : 3}})['py'])
    assert isinstance(copied, PyFileGenerator)
    assert copied is not generator
    assert copied.template_text == generator.template_text
    assert copied.relative_out_path() == 'updated/copied.py'
    assert generator.relative_out_path() == 'nested/py/testing.py'
    assert file_path.read_text(encoding='utf-8') == 'value = 3'

def test_serialize_roundtrip(tmp_path):
    generator = PyFileGenerator('testing.py', template_text='value = {value}')
    serialized = generator.serialize()
    copied = PyFileGenerator.unzerialize(serialized)
    file_path = Path(copied(tmp_path, {'py' : {'value' : 4}})['py'])
    assert serialized == {
        'template_text' : 'value = {value}',
        'out_file' : 'testing.py'
    }
    assert isinstance(copied, PyFileGenerator)
    assert copied.relative_out_path() == './testing.py'
    assert file_path.read_text(encoding='utf-8') == 'value = 4'

def test_no_params(tmp_path):
    template = "x = {{ 'a' : 1 }}"
    expected = "x = { 'a' : 1 }"
    generator = PyFileGenerator('testing.py', template_text=template)
    file_path = Path(generator(tmp_path, {}, tagged=False))
    assert file_path.read_text(encoding='utf-8') == expected
    file_path = Path(generator(tmp_path, {'py' : {}})['py'])
    assert file_path.read_text(encoding='utf-8') == expected
