import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Normal
from torch.autograd import Variable
import numpy as np
import random 

class Dynamics(nn.Module):

    def __init__(self, args):
        super().__init__()
        hidden_dim = args.hidden_dim
        self.args = args
        self.output_dim =  (args.observation_shape + 1)
        self.model = nn.Sequential(
            nn.Linear(args.input_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,  2* self.output_dim)
            )

        self.max_logvar = Variable(torch.ones((1, self.output_dim)).type(torch.FloatTensor) / 2, requires_grad=True).to(args.device)
        self.min_logvar = Variable(-torch.ones((1, self.output_dim)).type(torch.FloatTensor) * 10, requires_grad=True).to(args.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.model_lr)


    def forward(self, state, action, return_logvar = False):
        """
        Returns mean/logvar of delta_state/reward 
        """
        if not isinstance(state, torch.Tensor):
            state,action = torch.Tensor(state).unsqueeze(0).to(self.args.device), torch.Tensor(action).unsqueeze(0).to(self.args.device)
        concat = torch.cat((state,action), dim=1).to(self.args.device)

        output = self.model(concat)
        mean = output[:, :self.output_dim]

        logvar = self.max_logvar - F.softplus(self.max_logvar - output[:, self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if return_logvar:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def loss(self, mean, logvar, labels, inc_var_loss = True):

        inv_var = torch.exp(-logvar)

        if inc_var_loss:
            mse_loss = torch.mean(torch.pow(mean - labels , 2) * inv_var)
            var_loss = torch.mean(logvar)
            total_loss = mse_loss + var_loss 
        else:
            mse_loss = nn.MSELoss()
            total_loss = mse_loss(mean, labels)
        return total_loss 

    def train(self, loss):
        self.optimizer.zero_grad()
        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        loss.backward()
        self.optimizer.step()

    def sample(self, mean, logvar):
        with torch.no_grad():
            std = torch.sqrt(logvar)
            dist = Normal(mean, std)
            sample = dist.rsample()
        return sample

class Ensemble:
    def __init__(self, args):
        self.args = args 
        self.models = [Dynamics(args) for _ in range(args.ensemble_size)]


    def predict(self, state, action, one_model = False):
        
        if not isinstance(state, torch.Tensor):
            state,action = torch.Tensor(state).unsqueeze(0).to(self.args.device), torch.Tensor(action).unsqueeze(0).to(self.args.device)

        input_shape = state.shape[0]
        assert len(state.shape) == 2 & len(action.shape) == 2

        if not one_model:
            output_dim = self.args.observation_shape + 1
            ensemble_mean = np.zeros((self.args.ensemble_size, input_shape, output_dim))
            ensemble_logvar = np.zeros((self.args.ensemble_size, input_shape, output_dim))

            for i in range(self.args.ensemble_size):
                cur_mean, cur_logvar = self.models[i](state,action)
                assert cur_mean.shape[0] == input_shape
                ensemble_mean[i] = cur_mean.detach().cpu().numpy()
                ensemble_logvar[i] = cur_logvar.detach().cpu().numpy()

            return ensemble_mean, ensemble_logvar

        else:
            random_model = random.choice(self.models)
            mean, var = random_model(state, action)

            return mean.detach().cpu().numpy(), var.detach().cpu().numpy()


        

    def fit(self, state, action, labels, logger):
        """ 
        Arguments:
            inputs -> (state,action)
            labels -> torch.concat(delta state, reward)
        Called every loop of training, with input (batch_size, N)
        """
        for model in self.models:
            cur_mean, cur_logvar = model(state, action, return_logvar = True)
            loss = model.loss(cur_mean, cur_logvar, labels)
            model.train(loss)
            pred = model.sample(cur_mean , torch.exp(cur_logvar))
            print("Model Training, Loss:",loss.detach().item())
            logger.log("Model Train Loss", loss.detach().item())
            logger.log("Model Train pred_delta", pred[:,:-1])
            logger.log("Model Train pred_reward", pred[:,-1])

        logger.log("Model Train target_delta", labels[:,:-1])
        logger.log("Model Train target_reward", labels[:,-1])

    def eval(self, state, action, labels, logger):

        for model in self.models:
            cur_mean, cur_logvar = model(state, action) #This is actually not logvar 
            with torch.no_grad():
                loss = model.loss(cur_mean, cur_logvar, labels)
            pred = model.sample(cur_mean, cur_logvar)
            print("Model Evaluating, Loss:", loss)
            logger.log("Model Eval Loss", loss)
            logger.log("Model Eval pred_delta", pred[:,:-1])
            logger.log("Model Eval pred_reward", pred[:,-1])

        logger.log("Model Eval target_delta", labels[:,:-1])
        logger.log("Model Eval target_reward", labels[:,-1])










