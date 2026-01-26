from pathlib import Path
from daisypy.optim.dai_file_generator import DaiFileGenerator

EXPECTED = """(deffunction f Python
  "Call Python function."
  (module "testing")
  (name "linear")
  (domain [])
  (range []))

(defprogram print_it write
  "Write specific value"
  (declare v1 Number [] "V1")
  (declare v2 Number [] "V2")
  (v1 (apply f 0 []))
  (v2 (apply f 5 []))
  (what "f(0) = ${v1}, f(5) = ${v2}"))
  
(run print_it)
"""

def test_dai_file_generator(tmp_path):
    template_path = Path(__file__).parent / 'templates' / 'template.dai'
    generator = DaiFileGenerator('linear.dai', template_file_path=template_path)
    file_path = Path(generator(tmp_path, {'x1' : 0, 'x2' : 5})['dai'])
    assert file_path.read_text() == EXPECTED

    
    # lines_read = 0
    # with open(file_path, 'r', encoding='utf-8') as f:
    #     for i, line in enumerate(f):
    #         lines_read += 1
    #         if i == 4:
    #             assert line.strip() == "f(0) = 10, f(5) = 12.5"
    # assert lines_read == 6
    
    
