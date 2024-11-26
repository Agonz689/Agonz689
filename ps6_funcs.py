# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:57:17 2023

@author: ywang11
"""

import sklearn.datasets
import sklearn.model_selection
import sklearn.feature_selection
import sklearn.pipeline
import torch
import torch.nn
import torch.nn.functional
import skorch
import numpy as np

class PS6RegressorModule(torch.nn.Module):
    def __init__(
            self,
            n_input_features: int,
            dense_layer_sizes=(100, 100, ),  # only for hidden layers
            act_fn_dense=torch.nn.functional.relu,
            dropout=0.5
    ):
        super().__init__()
        self.n_input_features = n_input_features
        self.dense_layer_sizes = dense_layer_sizes
        self.act_fn_dense = act_fn_dense

        # Dense network after Conv network
        self.dense_network = torch.nn.ModuleList()
        self.dense_network.append(
            torch.nn.Linear(
                n_input_features,
                self.dense_layer_sizes[0])
        )
        # TODO: add dropout
        
        # Rest of the Dense network
        for layer_size in self.dense_layer_sizes[1:]:
            self.dense_network.append(
                torch.nn.Linear(
                    layer_size,
                    layer_size
                )
            )

        # Output layer
        self.output = torch.nn.Linear(self.dense_layer_sizes[-1], 1)


    def forward(self, X, **kwargs):
        for layer in self.dense_network:
            X = self.act_fn_dense(layer(X))
            
        return self.output(X)
    
    
    
    
    
    
    
    
    
    
    
    