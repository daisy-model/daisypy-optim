import tempfile
import os
from daisypy.optim import DefaultLogger

def test_default_logger(capsys):
    expected_out = 'line 1 info'
    expected_err = '\n'.join([
        'line 1 warning',
        'line 1 error'
    ])
    expected_result = [
        'step,tag,value,p1,p2,p3',
        '1,"a",0.1,0,2,4'
    ]
    with tempfile.TemporaryDirectory() as out_dir:
        expected_result_path = os.path.join(out_dir, 'result.csv')
        with DefaultLogger(out_dir) as logger:
            logger.info('line 1 info')
            logger.warning('line 1 warning')
            logger.error('line 1 error')
            logger.result(step=1, tag='a', value=0.1, p1=0, p2=2, p3=4)

        assert os.path.exists(expected_result_path)
        with open(expected_result_path, 'r', encoding='utf-8') as in_file:
            lines = [line.strip() for line in in_file]
        assert lines == expected_result

    captured = capsys.readouterr()
    assert captured.out.strip() == expected_out
    assert captured.err.strip() == expected_err
