from pathlib import Path
from daisypy.optim.py_file_generator import PyFileGenerator

EXPECTED = """def linear(x):
    return 0.5 * x + 10
"""

def test_py_file_generator(tmp_path):
    template_path = Path(__file__).parent / 'templates' / 'template.py'
    generator = PyFileGenerator('testing.py', template_file_path=template_path)
    file_path = Path(generator(tmp_path, {'a' : 0.5, 'b' : 10})['py'])
    assert file_path.read_text() == EXPECTED
    
    
