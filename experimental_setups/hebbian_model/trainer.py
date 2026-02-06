import numpy as np
import torch
import argparse

from typing import Dict


class Optimiser:
    """custom optimiser for SGD + Hebbian training updates"""

    def __init__(self, args: argparse.Namespace, perform_sgd=True):
        """Constructor for optimiser

        Args:
            args (argparse.Namespace): training params specified in parameters.py
        """
        self.lrate_sgd = args.lrate_sgd
        self.lrate_hebb = args.lrate_hebb
        self.hebb_normaliser = args.hebb_normaliser
        self.perform_sgd = args.perform_sgd
        self.perform_hebb = args.perform_hebb
        self.gating = args.gating
        self.losstype = args.loss_funct
        self.n_features = args.n_features
        self.n_hidden = args.n_hidden
        self.n_layers = args.n_layers
        self.ctx_twice = args.ctx_twice

    def step(self, model: torch.nn.Module, x_in: torch.Tensor, r_target: torch.Tensor):
        """a single training step, using procedure specified in args

        Args:
            model (torch.nn.Module): feed forward neural network
            x_in (torch.Tensor): training inputs
            r_target (torch.Tensor): training targets
        """

        if self.perform_sgd is True:
            self._sgd_update(model, x_in, r_target)
        if self.perform_hebb is True:
            if self.n_layers == 1:
                if self.gating == "oja":
                    self._oja_update(model, x_in)
                elif self.gating == "oja_ctx":
                    self._oja_ctx_update(model, x_in)
            elif self.n_layers == 2:
                if self.gating == "oja_ctx":
                    self._oja_ctx_update_2hidden(model, x_in)
                elif self.gating == "oja":
                    self._oja_update_2hidden(model, x_in)
                else:
                    raise NotImplementedError(
                        "Only oja_ctx or oja supported for 2 layer net"
                    )

    def _sgd_update(
        self, model: torch.nn.Module, x_in: torch.Tensor, r_target: torch.Tensor
    ):
        """performs stochastic gradient descent

        Args:
            model (torch.nn.Module): neural network
            x_in (torch.Tensor): training input data
            r_target (torch.Tensor): training labels
        """
        y_ = model(x_in)
        # compute loss
        loss = self.loss_funct(r_target, y_)
        # get gradients
        loss.backward()
        # update weights
        with torch.no_grad():
            for theta in model.parameters():
                if theta.requires_grad:
                    theta -= theta.grad * self.lrate_sgd
            model.zero_grad()

    def _oja_update(self, model: torch.nn.Module, x_in: torch.Tensor):
        """applies Oja's rule to weights from first hidden layer to second hidden layer

        Args:
            model (torch.nn.Module): feed forward neural network
            x_in (torch.Tensor): training data
        """
        x_vec = x_in.repeat(self.n_hidden).reshape(-1, self.n_features)

        with torch.no_grad():
            y = torch.t(model.W_h) @ x_in
            y = y.repeat(self.n_features).reshape(self.n_features, -1).T
            dW = self.lrate_hebb * y * (x_vec - y * torch.t(model.W_h))
            model.W_h += dW.T
            model.zero_grad()