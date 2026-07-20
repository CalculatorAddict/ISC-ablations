import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

import data
import numpy as np
import random
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
import utils
import sys
from ..isc_model.model import BCEMetric

class ErrorGatingModel(nn.Module):
    """
    Creates a Controlled Semantic Cognition (CSC) Model with error gating and Hebbian learning.

    Parameters
    ----------
    num_objects (int): Number of objects/inputs for the model. Default: 350
    num_hub_hidden_units (int): Number of hidden units in the hub layer. Default: 64
    num_context_dependent_hidden_units (int): Number of hidden units in the context-dependent layer. Default: 128
    num_task_context_units (int): Number of hidden units in the task context layer. Default: 16
    num_output (int): Number of output units. Default: 2541+2+3+350
    num_tasks (int): Number of tasks. Default: 36
    lr (float): Learning rate for the model. Default: 0.05
    device (str): Device to use for training. Default: None
    biases (bool): If ``True``, uses biases in the model. Default: True

    Attributes
    ----------
    item_input_to_hub_weights (torch.nn.Linear): Weights from the item input to the hub.
    context_input_to_task_context_rep_weights (torch.nn.Linear): Weights from the context input to the task context layer.
    task_context_rep_to_context_dependent_rep_weights (torch.nn.Linear): Weights from the task context layer to the context dependent layer.
    hub_to_context_dependent_rep_weights (torch.nn.Linear): Weights from the hub to the context dependent layer.
    context_dependent_rep_to_output_weights (torch.nn.Linear): Weights from the context dependent layer to the output layer.
    hub_to_output_weights (torch.nn.Linear): Weights from the hub to the output layer.
    loss_fn (torch.nn.BCEWithLogitsLoss): Loss function for the model.
    optimizer (torch.optim.Adam): Optimizer for the model.
    metrics (list): List of metrics to track during training.
    num_objects (int): Number of objects/inputs for the model.
    num_tasks (int): Number of tasks for the model.
    num_context_dependent_hidden_units (int): Number of hidden units in the context-dependent layer.
    device (str): Device to use for training.

    Methods
    -------
    freeze_weights()
        Freezes the weights of the model.
    load_old_model_weights(state_dict,use_old_size_starting_point=True)
        Loads weights from a previous model.
    get_context_independent_rep(x)
        Gets the context-independent representation of the model for a given input.
    get_task_context_rep(x)
        Gets the task context representation of the model for a given input.
    get_context_dependent_rep(x)
        Gets the context-dependent representation of the model for a given input.
    forward(x,take_sigmoid=True)
        Forward pass of the model.
    train(x,y,epochs=1,batch_size=64)
        Trains the model.
    plot_metrics()
        Plots the metrics of the model.
    get_task_context_reps()
        Gets the task context representations of the model for all input combinations.
    get_context_independent_reps()
        Gets the context-independent representations of the model for all input combinations.
    get_context_dependent_reps()
        Gets the context-dependent representations of the model for all input combinations.
    """
    def __init__(self, num_objects: int = 350, num_hub_hidden_units: int = 64,
                 num_context_dependent_hidden_units: int = 128,
                 num_output: int = 2541+2+3+350,
                 num_tasks: int = 36, lr: float = .05, lr_hebb: float = .001,
                 device=None, biases: bool = True,
                 has_hebbian_weight_updates: bool = True,
                 init_context = 0) -> None:
        super().__init__()

        if device is None:
            device = utils.set_torch_device()
        device = torch.device(device)

        self.lr = lr
        self.lr_hebb = lr_hebb # learning rate for Hebbian weight updates
        self.has_hebbian_weight_updates = has_hebbian_weight_updates
        self.register_buffer(
            "context",
            torch.tensor([[float(init_context)]], dtype=torch.float, device=device),
        )

        self.item_input_to_hub_weights = nn.Linear(num_objects,num_hub_hidden_units,device=device,bias=biases)
        self.context_input_to_context_dependent_rep_weights = nn.Linear(1,num_context_dependent_hidden_units,device=device,bias=False)
        self.hub_to_context_dependent_rep_weights = nn.Linear(num_hub_hidden_units,num_context_dependent_hidden_units,device=device,bias=biases)
        self.context_dependent_rep_to_output_weights = nn.Linear(num_context_dependent_hidden_units,num_output,device=device,bias=biases)
        self.hub_to_output_weights = nn.Linear(num_hub_hidden_units,num_output,device=device)
        
        nn.init.uniform_(self.item_input_to_hub_weights.weight,a=-.01,b=.01)
        nn.init.uniform_(self.context_input_to_context_dependent_rep_weights.weight,a=-.01,b=.01)
        nn.init.uniform_(self.hub_to_context_dependent_rep_weights.weight,a=-.01,b=.01)
        nn.init.uniform_(self.context_dependent_rep_to_output_weights.weight,a=-.01,b=.01)
        nn.init.uniform_(self.hub_to_output_weights.weight,a=-.01,b=.01)

        if biases:
            nn.init.uniform_(self.item_input_to_hub_weights.bias,a=-.01,b=.01)
            nn.init.uniform_(self.hub_to_context_dependent_rep_weights.bias,a=-.01,b=.01)
            nn.init.uniform_(self.context_dependent_rep_to_output_weights.bias,a=-.01,b=.01)
            nn.init.uniform_(self.hub_to_output_weights.bias,a=-.01,b=.01)
        else:
            with torch.no_grad():
                self.hub_to_output_weights.bias.copy_(torch.ones(self.hub_to_output_weights.bias.shape,device=device)*-2)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self._configure_trainable_parameters()
        self.optimizer = optim.Adam(self._trainable_parameters(), lr=lr)
        self.metrics = [BCEMetric()]
        self.context_input_to_cd_weight_history = []
        self.num_objects = num_objects
        self.num_tasks = num_tasks
        self.num_context_dependent_hidden_units = num_context_dependent_hidden_units
        self.device = device


    def freeze_weights(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def _configure_trainable_parameters(self) -> None:
        readout_parameter_names = {
            'context_dependent_rep_to_output_weights.weight',
            'context_dependent_rep_to_output_weights.bias',
            'hub_to_output_weights.weight',
            'hub_to_output_weights.bias',
        }

        for name, param in self.named_parameters():
            param.requires_grad = name in readout_parameter_names

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [param for param in self.parameters() if param.requires_grad]

    def load_old_model_weights(self, state_dict: dict) -> None:
        model_state = self.state_dict()
        for name, param in state_dict.items():
            if name in model_state:
                model_state[name].copy_(param)

        self._configure_trainable_parameters()
        self.optimizer = optim.Adam(self._trainable_parameters(), lr=self.lr)

    def get_context_independent_rep(self, x: torch.Tensor) -> torch.Tensor:
        hub_rep = torch.sigmoid(self.item_input_to_hub_weights(x[0]))
        return hub_rep


    def get_context_dependent_rep(self, x: torch.Tensor) -> torch.Tensor:
        hub_rep = self.get_context_independent_rep(x)
        task_context_rep = self.get_task_context_rep(x)
        item_in_context_rep = torch.sigmoid(self.task_context_rep_to_context_dependent_rep_weights(task_context_rep)+
                                            self.hub_to_context_dependent_rep_weights(hub_rep))
        return item_in_context_rep


    def forward(self, x: torch.Tensor, take_sigmoid: bool=True, noise: float=0) -> torch.Tensor:
        x = [t.to(self.device) for t in x]

        hub_rep = torch.sigmoid(self.item_input_to_hub_weights(x[0]))
        context = self._context_for_batch(x[0].size(0))
        item_in_context_rep = torch.sigmoid(self.context_input_to_context_dependent_rep_weights(context)+
                                            self.hub_to_context_dependent_rep_weights(hub_rep))
        output = self.hub_to_output_weights(hub_rep)+\
                 self.context_dependent_rep_to_output_weights(item_in_context_rep)
        if take_sigmoid:
            output = torch.sigmoid(output)

        # update context with error gating
        self._update_context(x, output.detach())

        return output


    # def train(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 1, batch_size: int = 1) -> list:
    #     if epochs < 1:
    #         return [0]
    #     for metric in self.metrics:
    #         metric(self(x),y,self)
    #     for epoch in range(epochs):
    #         n_steps = 0
    #         batch_idxs = np.random.permutation(range(len(y)))
    #         for batch_start in range(0,len(y),batch_size):
    #             batch_idx = batch_idxs[batch_start:min(batch_start+batch_size,len(y))]
    #             self.optimizer.zero_grad()
    #             output = self([x[0][batch_idx],x[1][batch_idx]],take_sigmoid=False)
    #             loss = self.loss_fn(output,y[batch_idx])
    #             loss.backward()
    #             self.optimizer.step()
    #             n_steps += 1

    #             if self.has_hebbian_weight_updates:
    #                 # Hebbian update with Oja's rule
    #                 self._oja_update([x[0][batch_idx],x[1][batch_idx]],normalize_update = True if batch_size > 1 else False)
    #                 if batch_size != 1:
    #                     raise ValueError("Oja update requires online updating after each iteration!")


    #         for metric in self.metrics:
    #             metric(self(x),y,self)
    #     return self.metrics
    

    def train(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 1, batch_size: int = 64, is_blocked: bool = False) -> list:
        x = [t.to(self.device) for t in x]
        y = y.to(self.device)
        if epochs < 1:
            return [0]
        for metric in self.metrics:
            metric(self(x),y,self)
        self.context_input_to_cd_weight_history = []
        self._record_context_input_to_cd_weights()

        for epoch in range(epochs):
            self.epoch = epoch
            n_steps = 0
            x_train, y_train, = self._train_load(x, y, is_blocked)

            for batch_start in range(0,len(y),batch_size):
                batch_end = min(batch_start + batch_size, len(y))
                batch_idx = range(batch_start,batch_end)

                self.optimizer.zero_grad()
                output = self([x_train[0][batch_idx],x_train[1][batch_idx]],take_sigmoid=True)

                # SGD update
                loss = self.loss_fn(output,y_train[batch_idx])
                loss.backward()
                self.optimizer.step()

                if self.has_hebbian_weight_updates:
                    # Hebbian update with Oja's rule
                    self._oja_update([x_train[0][batch_idx],x_train[1][batch_idx]],normalize_update = True if batch_size > 1 else False)

                n_steps += 1

            for metric in self.metrics:
                metric(self(x),y,self)
            self._record_context_input_to_cd_weights()
        return self.metrics
    
    def _train_load(self, x: torch.Tensor, y: torch.Tensor, is_blocked: bool):
        train_data = [x[0], x[1], y]

        # randomly permute training data
        if is_blocked:
            # if blocked, permute within each block then arrange blocks in sequence
            block_size = train_data[0].size(0)//2

            train_data_animal = [t[:block_size] for t in train_data]
            idxs_animal = torch.randperm(block_size, device=self.device)
            train_data_animal = [t[idxs_animal] for t in train_data_animal]

            train_data_instrument = [t[block_size:] for t in train_data]
            idxs_instrument = torch.randperm(block_size, device=self.device)
            train_data_instrument = [t[idxs_instrument] for t in train_data_instrument]

            # randomly pick which block goes first
            if random.random() < 0.5:
                train_data[0] = torch.cat((train_data_animal[0], train_data_instrument[0]))
                train_data[1] = torch.cat((train_data_animal[1], train_data_instrument[1]))
                train_data[2] = torch.cat((train_data_animal[2], train_data_instrument[2]))
            else:
                train_data[0] = torch.cat((train_data_instrument[0], train_data_animal[0]))
                train_data[1] = torch.cat((train_data_instrument[1], train_data_animal[1]))
                train_data[2] = torch.cat((train_data_instrument[2], train_data_animal[2]))

        else:
            idxs = torch.randperm(train_data[0].size(0), device=self.device)
            train_data = [t[idxs] for t in train_data]

        x_train = train_data[:2]
        y_train = train_data[2]

        return x_train, y_train
    

    def _oja_update(self, x, normalize_update: bool = False):
        """
        Apply Oja's update rule to the model's task representation to CD weights on input x.

        Parameters:
        - x (torch.Tensor): the input batch
        - normalize_update (bool): whether to apply spectral normalization to the update matrix
        """
        # w_item = self.hub_to_context_dependent_rep_weights.weight
        # w_task = self.context_input_to_task_context_rep_weights.weight

        # y_item = self.item_input_to_hub_weights(x[0])
        # y_task = self.context_input_to_task_context_rep_weights(x[1])

        w_task = self.context_input_to_context_dependent_rep_weights.weight

        task_rep = self._context_for_batch(x[0].size(0))
        y_task = self.context_input_to_context_dependent_rep_weights(task_rep)

        with torch.no_grad():
            # dw_item = (self.lr_hebb / x[0].size(0)) * torch.t(y_item) @ (x[0] - y_item @ w_item)
            # dw_task = (self.lr_hebb / x[1].size(0)) * torch.t(y_task) @ (x[1] - y_task @ w_task)

            # w_item.add_(dw_item)
            # w_task.add_(dw_task)
            if normalize_update:
                s = self._spectral_norm_power(y_task, n_iter=5)  # largest singular value
                lambda_max = (s * s) / y_task.size(0)            # batch size adjusted eigenvalue
                lr_hebb_eff = self.lr_hebb / (lambda_max + 1e-8) # norm-adjusted lr
            else:
                lr_hebb_eff = self.lr_hebb

            hebbian_term = torch.t(y_task) @ task_rep
            oja_decay = torch.sum(y_task.pow(2), dim=0, keepdim=True).t() * w_task
            dw_task = (lr_hebb_eff / task_rep.size(0)) * (hebbian_term - oja_decay)
            w_task.add_(dw_task)

        self.optimizer.zero_grad()

        # print("task_rep", task_rep.shape, "y_task", y_task.shape, "w_task", w_task.shape)


        if (torch.isnan(w_task.norm())):
            print(
                f"step={self.epoch} | "
                f"||w||={w_task.norm().item():.4e} | "
                f"||x||={task_rep.norm().item():.4e} | "
                f"y_max={y_task.abs().max().item():.4e} | "
                f"Δw_norm={dw_task.norm().item():.4e} | "
            )

            raise ValueError("w_task is nan")

    def _spectral_norm_power(self, Y, n_iter=5, eps=1e-8):
        # Y: (B, H) or (B, D) tensor
        v = torch.randn(Y.size(1), device=Y.device, dtype=Y.dtype)
        v = v / (v.norm() + eps)

        for _ in range(n_iter):
            v = Y.T @ (Y @ v)          # power iter on (Y^T Y)
            v = v / (v.norm() + eps)

        s = (Y @ v).norm()             # largest singular value of Y
        return s
    
    def _context_for_batch(self, batch_size: int) -> torch.Tensor:
        return self.context.detach().expand(batch_size, 1).clone()

    def _record_context_input_to_cd_weights(self) -> None:
        weights = self.context_input_to_context_dependent_rep_weights.weight.detach().flatten()
        self.context_input_to_cd_weight_history.append({
            'c1': float(weights[0].cpu().item()),
            'c2': float(weights[1].cpu().item()),
        })

    def _determine_true_label(self, x):
        """
        Determine true label for input to distractor task.

        Pre: x must have only one fully-activated one-hot input (ie there is exactly one idx s.t. x[0][idx]>=1.0)
        """

        x_item = x[0]
        if x_item.dim() == 1:
            x_item = x_item.unsqueeze(0)

        small_idxs = torch.tensor([105,115,29,57,59,262,253,266,267,260], device=x_item.device)
        large_idxs = torch.tensor([118,104,30,48,116,248,252,261,263,257], device=x_item.device)

        idx = torch.argmax(x_item, dim=1)

        labels = torch.full(idx.shape, torch.nan, dtype=torch.float, device=x_item.device)
        labels[torch.isin(idx, small_idxs)] = 0.0
        labels[torch.isin(idx, large_idxs)] = 1.0

        if torch.isnan(labels).any():
            raise ValueError("Input x did not have any recognized size-experiment item")
            
        return labels

        

    def _update_context(self, x, output):
        y_hat = (output[:, 2542] > output[:, 2541]).float()
        y_true = self._determine_true_label(x)

        error_signal = (y_hat - y_true).abs().mean()

        # switch contexts with probability given by error_signal
        if random.random() < error_signal:
            with torch.no_grad():
                self.context[0, 0] = 1.0 - self.context[0, 0]

    def plot_metrics(self) -> None:
        for metric in self.metrics:
            print(metric.name)
            if type(metric.values) is list:
                fig = px.line(y=metric.values)
            else:
                fig = px.line(metric.values,x='x',y='y',color='color')
            fig.show()


    def get_task_context_reps(self) -> np.array:
        item_x = torch.zeros((self.num_tasks,self.num_objects),device=self.device)
        context_x = torch.eye(self.num_tasks,device=self.device)
        tc_reps = self.get_task_context_rep([item_x,context_x]).cpu().detach().numpy()
        return tc_reps
    

    def get_context_independent_reps(self, indices = None) -> np.array:
        """
        Returns context-independent representations of the items corresponding to indices
        
        Parameters:
            indices (list): A list of relevant array indices, whose representations will be returned. Default is all indices.
        
        Returns a np.array with the embeddings.
        """
        if indices is None:
            indices = range(len(self.num_objects))
        item_x = torch.eye(self.num_objects,device=self.device)[indices]
        context_x = torch.zeros((len(indices),self.num_tasks),device=self.device)
        ind_reps = self.get_context_independent_rep([item_x,context_x]).cpu().detach().numpy()
        return ind_reps
    

    def get_context_dependent_reps(self, indices = None) -> np.array:
        if indices is None:
            indices = range(len(self.num_objects))
        item_x = torch.eye(self.num_objects,device=self.device)[indices]
        context_x = torch.zeros((len(indices),self.num_tasks),device=self.device)
        dep_reps = np.zeros((self.num_tasks, len(indices), self.num_context_dependent_hidden_units))
        for context in range(self.num_tasks):
            context_x = torch.zeros((len(indices),self.num_tasks),device=self.device)
            context_x[:,context] = 1
            dep_reps[context] = self.get_context_dependent_rep([item_x,context_x]).cpu().detach().numpy()
        return dep_reps
