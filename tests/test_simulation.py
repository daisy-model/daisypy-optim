# pylint: disable=missing-function-docstring
from pathlib import Path
import pytest
from daisypy.optim import DaiFileGenerator, OutputSpec, Simulation, StaticData


def test_rebases_environment_for_parent_relative_include(tmp_path, monkeypatch):
    layout_root = tmp_path / 'layout-root'
    layout_root.mkdir()
    monkeypatch.chdir(layout_root)

    source_dir = tmp_path / 'source'
    static_src = source_dir / 'common' / 'my-log.dai'
    static_src.parent.mkdir(parents=True)
    static_src.write_text('(dummy static file)', encoding='utf-8')

    output_dir = tmp_path / 'out'
    include_path = '../../common/my-log.dai'
    sim = Simulation(
        {
            'runfile' : DaiFileGenerator(
                'run.dai',
                template_text=f'(input file "{include_path}")',
                sub_dir='scenarios/site-1'
            )
        },
        {},
        [StaticData(static_src, Path('common'))]
    )

    run_path = sim.setup(output_dir, {'runfile' : {}})
    copied_static = output_dir / 'common' / 'my-log.dai'
    referenced_static = run_path.parent / include_path

    assert run_path == output_dir / 'scenarios' / 'site-1' / 'run.dai'
    assert copied_static.read_text(encoding='utf-8') == '(dummy static file)'
    assert include_path in run_path.read_text(encoding='utf-8')
    assert referenced_static.exists()
    assert referenced_static.read_text(encoding='utf-8') == '(dummy static file)'


def test_rebases_static_data_with_parent_relative_destination(tmp_path, monkeypatch):
    layout_root = tmp_path / 'layout-root'
    layout_root.mkdir()
    monkeypatch.chdir(layout_root)

    source_dir = tmp_path / 'source'
    static_src = source_dir / 'common' / 'my-log.dai'
    static_src.parent.mkdir(parents=True)
    static_src.write_text('(dummy static file)', encoding='utf-8')

    output_dir = tmp_path / 'out'
    include_path = '../../../common/my-log.dai'
    sim = Simulation(
        {
            'runfile' : DaiFileGenerator(
                'run.dai',
                template_text=f'(input file "{include_path}")',
                sub_dir='scenarios/site-1'
            )
        },
        {},
        [StaticData(static_src, Path('../common'))]
    )

    run_path = sim.setup(output_dir, {'runfile' : {}})
    copied_static = output_dir / 'common' / 'my-log.dai'
    referenced_static = run_path.parent / include_path

    assert run_path == output_dir / 'layout-root' / 'scenarios' / 'site-1' / 'run.dai'
    assert include_path in run_path.read_text(encoding='utf-8')
    assert copied_static.read_text(encoding='utf-8') == '(dummy static file)'
    assert referenced_static.exists()
    assert referenced_static.read_text(encoding='utf-8') == '(dummy static file)'


def test_rebases_output_paths_from_runfile_generator(tmp_path, monkeypatch):
    layout_root = tmp_path / 'layout-root'
    layout_root.mkdir()
    monkeypatch.chdir(layout_root)

    static_src = tmp_path / 'source' / 'common' / 'placeholder.txt'
    static_src.parent.mkdir(parents=True)
    static_src.write_text('placeholder', encoding='utf-8')

    sim = Simulation(
        {
            'runfile' : DaiFileGenerator(
                'run.dai',
                template_text='(run)',
                sub_dir='scenarios/site-1'
            )
        },
        {
            'field' : OutputSpec('field_water.dlf', 'water')
        },
        [StaticData(static_src, Path('common'))]
    )

    output_dir = tmp_path / 'out'
    sim.setup(output_dir, {'runfile' : {}})

    assert sim.outputs['field'].path() == (
        output_dir / 'scenarios' / 'site-1' / 'field_water.dlf'
    )


def test_simulation_raises_when_missing_runfile():
    with pytest.raises(ValueError, match="There must be a generated 'runfile'"):
        Simulation({
            'dai' : DaiFileGenerator('run.dai', template_text='(input file "path")'),
        }, {})
