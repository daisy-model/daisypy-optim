import subprocess
import pytest

def has_working_daisy():
    '''Check that daisy can be called and "works"'''
    # pylint: disable=broad-exception-caught
    try:
        return subprocess.run(["daisy", "-v"], check=False).returncode == 0
    except Exception:
        return False

requires_daisy = pytest.mark.skipif(
    not has_working_daisy(),
    reason="No working daisy installation found"
)
