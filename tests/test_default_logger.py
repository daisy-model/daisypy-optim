import os
from daisypy.optim import DefaultLogger

def test_default_logger(capsys, tmp_path):
    '''Test that DefaultLogger logs different messages to the expected locations'''
    expected_out = 'line 1 info'
    expected_err = '\n'.join([
        'line 1 warning',
        'line 1 error'
    ])
    expected_samples = [
        'step,tag,value,p1,p2,p3',
        '1,"a",0.1,0,2,4'
    ]
    expected_samples_path = tmp_path / 'samples.csv'
    expected_outcome = [
        'evaluation_id,time,predicted_value',
        '"eval-1","2000-01-01T00:00:00",0.1'
    ]
    expected_outcome_path = tmp_path / 'outcomes.csv'
    expected_target = [
        'objective_name,time,target_value',
        '"obj-1","2000-01-01T00:00:00",0.2'
    ]
    expected_target_path = tmp_path / 'targets.csv'
    with DefaultLogger(tmp_path) as logger:
        logger.info('line 1 info')
        logger.warning('line 1 warning')
        logger.error('line 1 error')
        logger.samples(step=1, tag='a', value=0.1, p1=0, p2=2, p3=4)
        logger.outcome(evaluation_id='eval-1', time='2000-01-01T00:00:00', predicted_value=0.1)
        logger.target(objective_name='obj-1', time='2000-01-01T00:00:00', target_value=0.2)

    assert os.path.exists(expected_samples_path)
    with open(expected_samples_path, 'r', encoding='utf-8') as in_file:
        lines = [line.strip() for line in in_file]
    assert lines == expected_samples
    assert os.path.exists(expected_outcome_path)
    with open(expected_outcome_path, 'r', encoding='utf-8') as in_file:
        lines = [line.strip() for line in in_file]
    assert lines == expected_outcome
    assert os.path.exists(expected_target_path)
    with open(expected_target_path, 'r', encoding='utf-8') as in_file:
        lines = [line.strip() for line in in_file]
    assert lines == expected_target

    captured = capsys.readouterr()
    assert captured.out.strip() == expected_out
    assert captured.err.strip() == expected_err
