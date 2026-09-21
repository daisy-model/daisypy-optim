# pylint: disable=missing-function-docstring,R0801
from pathlib import Path
import pytest
from daisypy.optim.dai_file_generator import DaiFileGenerator, SPAWN_PARALLEL_PARAM

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

def test_change_sub_dir(tmp_path):
    template_path = Path(__file__).parent / 'templates' / 'template.dai'
    generator = DaiFileGenerator(
        'linear.dai', template_file_path=template_path, sub_dir='nested/dai'
    )
    generator.sub_dir('dai')
    file_path = Path(generator(tmp_path, PARAMS))
    assert file_path == tmp_path / 'dai' / 'linear.dai'
    assert file_path.read_text(encoding='utf-8') == EXPECTED

def test_sub_dir_throws_when_abs():
    template_path = Path(__file__).parent / 'templates' / 'template.dai'
    with pytest.raises(ValueError, match='not relative'):
        generator = DaiFileGenerator(
            'linear.dai', template_file_path=template_path, sub_dir='/abs/path'
        )
    generator = DaiFileGenerator(
        'linear.dai', template_file_path=template_path, sub_dir='rel/path'
    )
    with pytest.raises(ValueError, match='not relative'):
        generator.sub_dir('/abs/path')

def test_relative_out_path():
    generator = DaiFileGenerator('linear.dai', template_text='(run test)', sub_dir='nested/dai')
    assert generator.relative_out_path() == Path('nested/dai/linear.dai')

def test_no_params(tmp_path):
    template = '(defprogram print_it write\n  (what "${{v1}}"))'
    expected = '(defprogram print_it write\n  (what "${v1}"))'
    generator = DaiFileGenerator('linear.dai', template_text=template)
    file_path = Path(generator(tmp_path, {}))
    assert file_path.read_text(encoding='utf-8') == expected


SPAWN_1 = """(defprogram p1 spawn (program p2 p3) (parallel 10))"""
SPAWN_2 = """(defprogram p1 spawn (program p2 p3))"""
SPAWN_P1 = """(defprogram p1 spawn\n  (program p2 p3) (parallel 1))"""
SPAWN_P3 = """(defprogram p1 spawn\n  (program p2 p3) (parallel 3))"""

def test_spawn_is_parameterized(tmp_path):
    generator = DaiFileGenerator('dummy', template_text=SPAWN_1)
    assert generator.has_spawn_program
    file_path = Path(generator(tmp_path, {SPAWN_PARALLEL_PARAM : 1}))
    assert file_path.read_text(encoding='utf-8') == SPAWN_P1
    file_path = Path(generator(tmp_path, {SPAWN_PARALLEL_PARAM : 3}))
    assert file_path.read_text(encoding='utf-8') == SPAWN_P3

    generator = DaiFileGenerator('dummy', template_text=SPAWN_2)
    assert generator.has_spawn_program
    file_path = Path(generator(tmp_path, {SPAWN_PARALLEL_PARAM : 1}))
    assert file_path.read_text(encoding='utf-8') == SPAWN_P1
    file_path = Path(generator(tmp_path, {SPAWN_PARALLEL_PARAM : 3}))
    assert file_path.read_text(encoding='utf-8') == SPAWN_P3
