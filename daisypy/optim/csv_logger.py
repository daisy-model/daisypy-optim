import os
from multiprocessing import Queue, Process
import numpy as np

def queue_writer(log_paths, queue):
    logs = {
        name : open(path, 'w', encoding='utf-8') for name, path in log_paths.items()
    }
    header = {
        'scalar' :'tag,step,value\n',
        'samples' : 'tag,parameter,step,value\n',
        'distributions' : 'tag,parameter,step,mean,std\n'
    }
    for k,v in header.items():
        logs[k].write(v)
        logs[k].flush()

    try:
        while True:
            rows, log_name = queue.get(block=True)
            logs[log_name].writelines((row + '\n' for row in rows))
    except Exception: # TypeError
        # The element in the queue wasnt a pair. This is used to signal that we are done
        for log in logs.values():
            log.flush()
            log.close()


class CsvLogger():
    """A simple csv logger. Use as context manager OR call close method explicityle OR ensure the
    object is destroyed (e.g. goes out of scope)
    """
    def __init__(self, logdir, tag, log_every_nth=1):
        self.queue = Queue()
        logdir = os.path.join(logdir, tag)
        os.makedirs(logdir, exist_ok=True)
        log_paths = {
            'scalar' : os.path.join(logdir, 'scalars.csv'),
            'samples' : os.path.join(logdir, 'samples.csv'),
            'distributions' : os.path.join(logdir, 'distributions.csv')
        }
        self.writer = Process(target=queue_writer, args=(log_paths, self.queue,))
        self.writer.start()

        if not isinstance(log_every_nth, dict):
            self.log_every_nth = {
                'scalar' : log_every_nth,
                'samples' : log_every_nth,
                'distributions' : log_every_nth
            }
        else:
            self.log_every_nth = log_every_nth

    def log_samples(self, tag, parameters, samples, step, **kwargs):
        if step % self.log_every_nth['samples'] != 0:
            return
        samples = np.array(samples)
        rows = []
        for j, param in enumerate(parameters):
            for i in range(len(samples)):
                rows.append(f'"{tag}","{param.name}",{step},{samples[i,j]}')
        self.queue.put((rows, 'samples'))

    def log_scalar(self, tag, value, step):
        if step % self.log_every_nth['scalar'] != 0:
            return
        rows = [f'"{tag}",{step},{value}']
        self.queue.put((rows, 'scalar'))

    def log_parameter_distributions(self, tag, parameters, means, stds, step, single_figure=True):
        if step % self.log_every_nth['distributions'] != 0:
            return
        rows = []
        for param, mean, std in zip(parameters, means, stds):
            rows.append(f'"{tag}","{param.name}",{step},{mean},{std}')
        self.queue.put((rows, 'distributions'))

    def close(self):
        print('Closing')
        self.queue.put(None)
        self.writer.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('Exiting')
        self.close()

    def __del__(self):
        print('Deleting')
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
