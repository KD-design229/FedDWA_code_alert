import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader, Subset

class ALA:
    def __init__(self,
                cid: int,
                loss: nn.Module,
                train_data, 
                batch_size: int, 
                rand_percent: int, 
                layer_idx: int = 0,
                eta: float = 1.0,
                device: str = 'cpu', 
                threshold: float = 0.1,
                num_pre_loss: int = 10) -> None:
        """
        Initialize ALA module
        """

        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.weights = None # Learnable local aggregation weights.
        self.start_phase = True


    def adaptive_local_aggregation(self, 
                            global_model,
                            local_model: nn.Module) -> None:
        """
        Generates the Dataloader for the randomly sampled local training data and 
        preserves the lower layers of the update. 
        """

        # randomly sample partial local training data
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio*len(self.train_data))
        if rand_num == 0:
            return
            
        rand_idx = random.randint(0, len(self.train_data)-rand_num)
        
        # Use Subset for safe slicing of Datasets
        indices = list(range(rand_idx, rand_idx+rand_num))
        subset = Subset(self.train_data, indices)
        rand_loader = DataLoader(subset, self.batch_size, drop_last=False)

        # obtain the references of the parameters
        params = list(local_model.parameters())
        
        # [Fix] Handle global_model being a dict (state_dict) or nn.Module
        params_gp = []
        params_p = []
        params_tp = []
        
        # temp local model only for weight learning
        model_t = copy.deepcopy(local_model)
        params_t_dict = dict(model_t.named_parameters())

        if isinstance(global_model, dict):
            # global_model is a state_dict (likely only containing trainable params)
            for name, param in local_model.named_parameters():
                if name in global_model and param.requires_grad:
                    params_p.append(param)
                    params_gp.append(global_model[name])
                    params_tp.append(params_t_dict[name])
        else:
            # global_model is nn.Module
            params_g = list(global_model.parameters())
            # deactivate ALA at the 1st communication iteration
            if torch.sum(params_g[0] - params[0]) == 0:
                return
            
            trainable_indices = [i for i, p in enumerate(params) if p.requires_grad]
            if not trainable_indices:
                return
                
            params_p = [params[i] for i in trainable_indices]
            params_gp = [params_g[i] for i in trainable_indices]
            params_tp = [list(model_t.parameters())[i] for i in trainable_indices]

        if not params_p:
            return

        # frozen the non-trainable layers in temp model to reduce computational cost
        # (Actually they are already frozen in model_t if copied from local_model)
        
        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # initialize the weight to all ones in the beginning
        if self.weights is None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,
                                                self.weights):
            param_t.data = param + (param_g - param) * weight

        # weight learning
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        while True:
            for x, y in rand_loader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                optimizer.zero_grad()
                output = model_t(x)
                loss_value = self.loss(output, y) # modify according to the local objective
                loss_value.backward()

                # update weight in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    # weight = weight - eta * grad * (param_g - param)
                    # We clamp weights to [0, 1] to keep interpolation valid
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    param_t.data = param + (param_g - param) * weight

            losses.append(loss_value.item())
            cnt += 1

            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                break

            # train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                # print('Client:', self.cid, '\tStd:', np.std(losses[-self.num_pre_loss:]),
                #     '\tALA epochs:', cnt)
                break

        self.start_phase = False

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()
