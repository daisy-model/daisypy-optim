from daisypy.optim.terminal_log import TerminalLog

def test_terminal_log(capsys):
    expected_out = '\n'.join([
        'tag=test,step=1,value=0.1,message=string',
        'test,2,0.2,string 2',
    ])
    expected_err = '\n'.join([
        'done,step=3,value=0.3',
    ])
    with TerminalLog() as info, TerminalLog(error=True) as error:
        info.log(tag='test', step=1, value=0.1, message="string")
        info.log('test', 2, 0.2, "string 2")
        error.log('done', step=3, value=0.3)

    captured = capsys.readouterr()
    assert captured.out.strip() == expected_out
    assert captured.err.strip() == expected_err
