from daisypy.optim.csv_log import CsvLog

def test_csv_log(tmp_path):
    '''Test that CsvLog logs the expected messages and handles cleanup nicely'''
    expected = [
        'tag,step,value,message',
        'test,1,0.1,"quoted string"',
        'test,2,0.2,"quoted string 2"',
        'done,3,0.3,"quoted string 3"',
    ]
    columns = {
        'tag' : str,
        'step' : lambda x : str(int(x)),
        'value' : lambda x : str(float(x)),
        'message' : None
    }
    path = tmp_path / 'test-log.csv'
    with CsvLog(path, columns) as log:
        log.log(tag='test', step=1, value=0.1, message="quoted string", flush=False)
        log.log(tag='test', step=2, value=0.2, message="quoted string 2", flush=True)
        with open(path, 'r', encoding='utf-8') as infile:
            lines = [line.strip() for line in infile]
        assert lines == expected[:-1]
        log.log(tag='done', step=3, value=0.3, message="quoted string 3", flush=False)

    # Should be flushed when closed
    with open(path, 'r', encoding='utf-8') as infile:
        lines = [line.strip() for line in infile]
    assert lines ==  expected
