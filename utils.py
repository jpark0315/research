
import random
import numpy as np
from collections import OrderedDict

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def normalize(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        #batch_count = x.shape[0]
        batch_count = 1 #HACKED
        self.update_from_moments(batch_mean, batch_var, batch_count)
        return np.clip((x - self.mean)/np.sqrt(self.var), -5, 5)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count /(self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count



class Logger:
    def __init__(self):

        self.dict = OrderedDict()

    def log(self, name, data):
        if name not in self.dict.keys():
            self.dict[name] =  [data]

        else:
            self.dict[name].append(data)

    def plot(self):

        import matplotlib.pyplot as plt
        import torch 
        for k,v in self.dict.items():
            if not isinstance(v[0], torch.Tensor):
                plt.figure()
                if k == 'Dones' or k == "pred_values reward" or k == "real reward":
                    print('MEAN/VAR', np.asarray(v).mean(), np.asarray(v).std())
                plt.title(k)
                plt.plot(v)
                
                plt.show()
                
                  
            else:
                data = torch.stack(v)
                if len(data) == 1:
                    plt.figure()
                    plt.title(k)
                    plt.plot(data.mean(1))
                else:
                    plt.figure()
                    plt.title(k)
                    plt.plot(data)
                
