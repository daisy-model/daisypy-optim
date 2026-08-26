# pylint: disable=missing-function-docstring,R0801
from pathlib import Path
import pytest
from daisypy.optim.dai_file_generator import DaiFileGenerator

EXPECTED = """(deffunction f Python
  "Call Python function." (module "testing") (name "linear") (domain []) (range []))
(defprogram print_it write
  "Write specific value"
  (declare v1 Number [] "V1")
  (declare v2 Number [] "V2")
  (v1 (apply f 0 []))
  (v2 (apply f 5 []))
  (what "f(0) = ${v1}, f(5) = ${v2}"))
(run print_it)"""

PARAMS = { 'x1' : 0, 'x2' : 5 }

def test_tagged(tmp_path):
    '''Test that generated dai file is as expected'''
    template_path = Path(__file__).parent / 'templates' / 'template.dai'
    generator = DaiFileGenerator('linear.dai', template_file_path=template_path)
    file_path = Path(generator(tmp_path, PARAMS))
    assert file_path.read_text(encoding='utf-8') == EXPECTED

def test_generate_from_plain_params(tmp_path):
    template_path = Path(__file__).parent / 'templates' / 'template.dai'
    generator = DaiFileGenerator('linear.dai', template_file_path=template_path)
    file_path = Path(generator(tmp_path, PARAMS))
    assert file_path.read_text(encoding='utf-8') == EXPECTED

def test_sub_dir(tmp_path):
    template_path = Path(__file__).parent / 'templates' / 'template.dai'
    generator = DaiFileGenerator(
        'linear.dai', template_file_path=template_path, sub_dir='nested/dai'
    )
    file_path = Path(generator(tmp_path, PARAMS))
    assert file_path == tmp_path / 'nested' / 'dai' / 'linear.dai'
    assert file_path.read_text(encoding='utf-8') == EXPECTED

def test_relative_out_path():
    generator = DaiFileGenerator('linear.dai', template_text='(run test)', sub_dir='nested/dai')
    assert generator.relative_out_path() == 'nested/dai/linear.dai'

def test_copy_and_update(tmp_path):
    generator = DaiFileGenerator('linear.dai', template_text='(run {value})', sub_dir='nested/dai')
    copied = generator.copy_and_update(out_file='copied.dai', sub_dir='updated')
    file_path = Path(copied(tmp_path, {'value' : 'test'}))
    assert isinstance(copied, DaiFileGenerator)
    assert copied is not generator
    assert copied.template_text == generator.template_text
    assert copied.relative_out_path() == 'updated/copied.dai'
    assert generator.relative_out_path() == 'nested/dai/linear.dai'
    assert file_path.read_text(encoding='utf-8') == '(run test)'

def test_no_params(tmp_path):
    template = '(defprogram print_it write\n  (what "${{v1}}"))'
    expected = '(defprogram print_it write\n  (what "${v1}"))'
    generator = DaiFileGenerator('linear.dai', template_text=template)
    file_path = Path(generator(tmp_path, {}))
    assert file_path.read_text(encoding='utf-8') == expected

SPAWN_PARALLEL = """(defprogram p1 spawn (parallel 10))"""
SPAWN_SEQUENTIAL = """(defprogram p1 spawn\n  (parallel 1))"""

def test_spawn_is_made_sequential():
    with pytest.warns(UserWarning, match="spawn forced to 1"):
        generator = DaiFileGenerator('dummy', template_text=SPAWN_PARALLEL)
    assert generator.template_text == SPAWN_SEQUENTIAL
