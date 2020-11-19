import random 
import numpy as np 


class ReplayBuffer:
    """
    Buffer to train model
    """
    def __init__(self, capacity, seed, args, true_buffer = False):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.eval_buffer = []
        self.position = 0
        self.eval_position = 0
        self.args = args
        self.true_buffer = true_buffer 
    def push(self, state, action, reward, next_state, done):
                          
        if np.random.uniform(0,1) < 0.9:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.capacity
        else:
            if len(self.eval_buffer) < self.capacity:
                self.eval_buffer.append(None)
            self.eval_buffer[self.eval_position] = (state,action, reward, next_state, done)
            self.eval_position = (self.eval_position + 1) % self.capacity

    def sample(self, batch_size, train = True):
        
        # if self.true_buffer:
        #     idx = int(len(self.buffer) - self.args.E_steps)
        #     if train:
        #         batch = random.sample(self.buffer[:idx], batch_size)
        #     else:
        #         #Only test on newly arrived samples
        #         batch = random.sample(self.buffer[idx:], batch_size)
        if self.true_buffer:
            if train:
                batch = random.sample(self.buffer, batch_size)
            
            else:
                batch = random.sample(self.eval_buffer, min(batch_size,len(self.eval_buffer)))

        else: 
            batch = random.sample(self.buffer + self.eval_buffer, batch_size)
    
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_state(self):
        """
        Samples random starting state for rollouts
        """
        position = random.choice(range(len(self.buffer)))
        return self.buffer[position][0]

    def __len__(self):
        return len(self.buffer)
