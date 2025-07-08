import os
import numpy as np

def write(rows, log):
    log.writelines((row + '\n' for row in rows))
    log.flush()
    os.fsync(log.fileno())

class CsvLogger():
    """A simple csv logger. Use as context manager OR call close method explicityle OR ensure the
    object is destroyed (e.g. goes out of scope)
    """
    def __init__(self, logdir, tag, log_every_nth=1):
        logdir = os.path.join(logdir, tag)
        os.makedirs(logdir, exist_ok=True)
        log_paths = {
            'scalar' : os.path.join(logdir, 'scalars.csv'),
            'samples' : os.path.join(logdir, 'samples.csv'),
            'distributions' : os.path.join(logdir, 'distributions.csv')
        }
        self.logs = {
            name : open(path, 'w', encoding='utf-8') for name, path in log_paths.items()
        }
        header = {
            'scalar' :'tag,step,value\n',
            'samples' : 'tag,parameter,step,value\n',
            'distributions' : 'tag,parameter,step,mean,std\n'
        }
        for k,v in header.items():
            self.logs[k].write(v)
            self.logs[k].flush()


        if not isinstance(log_every_nth, dict):
            self.log_every_nth = {
                'scalar' : log_every_nth,
                'samples' : log_every_nth,
                'distributions' : log_every_nth
            }
        else:
            self.log_every_nth = log_every_nth

    def log_samples(self, tag, parameters, samples, step, *args, **kwargs):
        if step % self.log_every_nth['samples'] != 0:
            return
        samples = np.array(samples)
        rows = []
        for j, param in enumerate(parameters):
            for i in range(len(samples)):
                rows.append(f'"{tag}","{param.name}",{step},{samples[i,j]}')
        write(rows, self.logs['samples'])

    def log_scalar(self, tag, value, step):
        if step % self.log_every_nth['scalar'] != 0:
            return
        rows = [f'"{tag}",{step},{value}']
        write(rows, self.logs['scalar'])

    def log_parameter_distributions(self, tag, parameters, means, stds, step, single_figure=True):
        if step % self.log_every_nth['distributions'] != 0:
            return
        rows = []
        for param, mean, std in zip(parameters, means, stds):
            rows.append(f'"{tag}","{param.name}",{step},{mean},{std}')
        write(rows, self.logs['distributions'])

    def close(self):
        for log in self.logs.values():
            log.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


def testit():
    class Dummy:
        def __init__(self, name):
            self.name = name

    logger = CsvLogger('tmp-log/csv', 'test')
    for i in range(10):
        logger.log_scalar('x^2', i**2, i)

    rng = np.random.default_rng()
    for i in [0, 5, 10]:
        samples = rng.random((10, 3))
        parameters = [ Dummy('a'), Dummy('b'), Dummy('c') ]
        logger.log_samples('ran', parameters, samples, i)
        logger.log_samples('ran_sep', parameters, samples, i)

if __name__ == '__main__':
    testit()
    # Or using `with`
    # with CsvLogger('tmp-log/csv', 'test') as logger:
    #     for i in range(10):
    #         logger.log_scalar('x^2', i**2, i)

    #     rng = np.random.default_rng()
    #     for i in [0, 5, 10]:
    #         samples = rng.random((10, 3))
    #         parameters = [ Dummy('a'), Dummy('b'), Dummy('c') ]
    #         logger.log_samples('ran', parameters, samples, i)
    #         logger.log_samples('ran_sep', parameters, samples, i)
