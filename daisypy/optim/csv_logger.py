import os
from multiprocessing import Queue, Process
import numpy as np

def queue_writer(queue):
    try:
        while True:
            strings, target = queue.get(block=True)
            with open(target, 'a') as out:
                print(*strings, sep='\n', file=out)
    except TypeError:
        # The element in the queue wasnt a pair. This is used to signal that we are done
        pass

class CsvLogger():
    def __init__(self, logdir, tag, log_every_nth=1):
        self.queue = Queue()
        self.writer = Process(target=queue_writer, args=(self.queue,))
        self.writer.start()
        logdir = os.path.join(logdir, tag)
        os.makedirs(logdir, exist_ok=True)
        self.log_paths = {
            'scalar' : os.path.join(logdir, 'scalars.csv'),
            'samples' : os.path.join(logdir, 'samples.csv'),
            'distributions' : os.path.join(logdir, 'distributions.csv')
        }
        with open(self.log_paths['scalar'], 'w', encoding='utf-8') as out:
            print('tag,step,value', file=out)
        with open(self.log_paths['samples'], 'w', encoding='utf-8') as out:
            print('tag,parameter,step,value', file=out)
        with open(self.log_paths['distributions'], 'w', encoding='utf-8') as out:
            print('tag,parameter,step,mean,std', file=out)

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
        self.queue.put((rows, self.log_paths['samples']))

    def log_scalar(self, tag, value, step):
        if step % self.log_every_nth['scalar'] != 0:
            return
        rows = [f'"{tag}",{step},{value}']
        self.queue.put((rows, self.log_paths['scalar']))

    def log_parameter_distributions(self, tag, parameters, means, stds, step, single_figure=True):
        if step % self.log_every_nth['distributions'] != 0:
            return
        rows = []
        for param, mean, std in zip(parameters, means, stds):
            rows.append(f'"{tag}","{param.name}",{step},{mean},{std}')
        self.queue.put((rows, self.log_paths['distributions']))

    def close(self):
        self.queue.put(None)
        self.writer.join()

    def __del__(self):
        self.close()

if __name__ == '__main__':
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
    logger.close()
