from pathlib import Path
from daisypy.optim import (
    PyFileGenerator, DaiFileGenerator, MultiFileGenerator, DaisyRunner
)

def test_run_with_python(tmp_path):
    EXPECTED = "f(4) = 8, f(-1) = 6.75"
    template_dir = Path(__file__).parent / 'templates'
    generator = MultiFileGenerator({
        'py' : PyFileGenerator('testing.py', template_file_path=template_dir / 'template.py'),
        'dai' : DaiFileGenerator('run.dai', template_file_path=template_dir / 'template.dai')
    })
    
    params = {
        'py' : { 'a' : 0.25, 'b' : 7 },
        'dai' : { 'x1' : 4, 'x2' : -1 }
    }

    # Assume we are on linux and daisy is installed
    runner = DaisyRunner('daisy')
    dai_file = generator(tmp_path, params)['dai']
    result = runner(dai_file, str(tmp_path))
    assert result.returncode == 0

    daisy_log = Path(tmp_path / 'daisy.log')
    with daisy_log.open(encoding='utf-8') as f:
        lines = [ line for line in f ]
    assert len(lines) >= 2
    assert lines[-2].strip() == EXPECTED
       
def test_run_with_several_python_files(tmp_path):
    tmp_path = Path('out')
    EXPECTED = "f(4) = 6, f(-1) = 7.25"
    template_dir = Path(__file__).parent / 'templates'
    python_dir = Path(__file__).parent / 'py-files'
    # 'py' generates the python module that is called from Daisy. Filename and function name must
    # match what is defined in the 'dai' file.
    # 'util' generates a python script that is imported by 'py'. Filename and function name must
    # match what is defined in the 'py' file.
    # 'dai' generates the file run by Daisy. You should only generate one dai file.
    generator = MultiFileGenerator({
        'py' : PyFileGenerator('testing.py', template_file_path=template_dir / 'template2.py'),
        'util' : PyFileGenerator('util.py', template_file_path=python_dir / 'util.py'),
        'dai' : DaiFileGenerator('run.dai', template_file_path=template_dir / 'template.dai')
    })
    
    params = {
        'py' : { 'a' : 0.25, 'b' : 7 },
        'dai' : { 'x1' : 4, 'x2' : -1 },
        'util' : {} # No parameters
    }

    # Assume we are on linux and daisy is installed
    runner = DaisyRunner('daisy')
    dai_file = generator(tmp_path, params)['dai']
    result = runner(dai_file, str(tmp_path))
    assert result.returncode == 0

    daisy_log = Path(tmp_path / 'daisy.log')
    with daisy_log.open(encoding='utf-8') as f:
        lines = [ line for line in f ]
    assert len(lines) >= 2
    assert lines[-2].strip() == EXPECTED
    

    
