import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, logdir, tag, log_every_nth=1):
        logdir = os.path.join(logdir, tag)
        self.writer = SummaryWriter(logdir)
        if not isinstance(log_every_nth, dict):
            self.log_every_nth = {
                'scalar' : log_every_nth,
                'samples' : log_every_nth,
                'distributions' : log_every_nth
            }
        else:
            self.log_every_nth = log_every_nth

    def log_samples(self, tag, parameters, samples, step, single_figure=True):
        if step % self.log_every_nth['samples'] != 0:
            return
        samples = np.array(samples)
        names = [p.name for p in parameters]
        if single_figure:
            figures, ax = plt.subplots()
            n_samples, n_params = samples.shape
            for i in range(n_params):
                y = np.full(n_samples, i)
                x = samples[:,i]
                ax.scatter(x, y)
            ax.set_yticks(range(n_params), names)
            figures.tight_layout()
        else:
            figures = []
            for i, name in enumerate(names):
                fig, ax = plt.subplots(figsize=(2,10))
                n_samples, n_params = samples.shape
                x = np.full(n_samples, i)
                y = samples[:,i]
                ax.scatter(x, y)
                ax.set_title(name)
                ax.set_xticks([])
                ax.tick_params(axis="y",direction="in", pad=-22)
                fig.tight_layout()
                figures.append(fig)                
        self.writer.add_figure(tag, figures, global_step=step)


    def log_scalar(self, tag, value, step):
        if step % self.log_every_nth['scalar'] != 0:
            return        
        self.writer.add_scalar(tag, value, step)    
        
    def log_parameter_distributions(self, tag, parameters, means, stds, step, single_figure=True):
        if step % self.log_every_nth['distributions'] != 0:
            return
        names = [p.name for p in parameters]
        if single_figure:
            low = np.min(means - 3*stds)
            high = np.max(means + 3*stds)
            x = np.linspace(low, high)
            figures, ax = plt.subplots()
            for mean, std in zip(means, stds):
                ax.plot(x, stats.norm.pdf(x, mean, std))
            ax.legend(names)
            figures.tight_layout()
        else:
            figures = []
            for param, mean, std in zip(parameters, means, stds):
                fig, ax = plt.subplots()
                x = np.linspace(mean - 3*std, mean + 3*std)
                ax.plot(x, stats.norm.pdf(x, mean, std))
                ax.axvline(param.valid_range[0])
                ax.axvline(param.valid_range[1])
                ax.set_title(param.name)
                fig.tight_layout()
                figures.append(fig)                                    
        self.writer.add_figure(tag, figures, global_step=step)

    def close(self):
        self.writer.close()


if __name__ == '__main__':
    class Dummy:
        def __init__(self, name):
            self.name = name
        
    logger = TensorBoardLogger('tmp-log', 'test')
    for i in range(10):
        logger.log_scalar('x^2', i**2, i)

    rng = np.random.default_rng()
    for i in [0, 5, 10]:
        samples = rng.random((10, 3))
        parameters = [ Dummy('a'), Dummy('b'), Dummy('c') ]
        logger.log_samples('ran', parameters, samples, i)
        logger.log_samples('ran_sep', parameters, samples, i, False)
    logger.close()
