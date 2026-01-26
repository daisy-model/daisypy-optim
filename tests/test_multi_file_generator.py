from pathlib import Path
from daisypy.optim import (
    PyFileGenerator, DaiFileGenerator, MultiFileGenerator
)

EXPECTED = {
    'py' : """def linear(x):
    return 0.25 * x + 7""",
    
    'dai' : """(deffunction f Python
  "Call Python function."
  (module "testing")
  (name "linear")
  (domain [])
  (range []))

(defprogram print_it write
  "Write specific value"
  (declare v1 Number [] "V1")
  (declare v2 Number [] "V2")
  (v1 (apply f 4 []))
  (v2 (apply f -1 []))
  (what "f(4) = ${v1}, f(-1) = ${v2}"))
  
(run print_it)"""
}

def test_multi_file_generators(tmp_path):
    template_dir = Path(__file__).parent / 'templates'
    generator = MultiFileGenerator({
        'py' : PyFileGenerator('testing.py', template_file_path=template_dir / 'template.py'),
        'dai' : DaiFileGenerator('run.dai', template_file_path=template_dir / 'template.dai')
    })
    
    params = {
        'py' : { 'a' : 0.25, 'b' : 7 },
        'dai' : { 'x1' : 4, 'x2' : -1 }
    }
    
    for gen_name, file_path in generator(tmp_path, params).items():
        assert Path(file_path).read_text().strip() == EXPECTED[gen_name]
