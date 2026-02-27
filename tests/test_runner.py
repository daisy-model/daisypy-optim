from pathlib import Path
from daisypy.optim import DaisyRunner
from .markers import requires_daisy

EXPECTED = "Hello from Daisy"

@requires_daisy
def test_runner(tmp_path):
    '''Test that DaisyRunner can run Daisy and generate the expected daisy.log'''
    # Assume we are on linux and daisy is installed
    runner = DaisyRunner('daisy')
    dai_path = Path(__file__).parent / 'hello.dai'
    result = runner(str(dai_path), str(tmp_path))
    assert result.returncode == 0
    
    daisy_log = tmp_path / 'daisy.log'
    with daisy_log.open(encoding='utf-8') as f:
        lines = list(f)
    assert len(lines) >= 2
    assert lines[-2].strip() == EXPECTED
