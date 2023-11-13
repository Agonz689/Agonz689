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

class ClayClassifierModule(torch.nn.Module):
    def __init__(
            self,
            n_input_channels: int,
            input_img_height: int,  # assume images are square
            n_output_probs: int,
            conv_layer_sizes=(32, 64, ),  # includes the first conv layer
            conv_kernel_sizes=(3, 3, ),  # includes the first conv layer
            act_fn_maxpool=torch.nn.functional.relu,
            dense_layer_sizes=(100, 100, ),  # only for hidden layers
            act_fn_dense=torch.nn.functional.relu,
            dropout=0.5
    ):
        super().__init__()
        self.input_img_height = input_img_height
        self.conv_layer_sizes = conv_layer_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.act_fn_maxpool = act_fn_maxpool
        self.dense_layer_sizes = dense_layer_sizes
        self.act_fn_dense = act_fn_dense
        
        # Conv network initialization
        self.conv_network = torch.nn.ModuleList()
        self.conv_network.append(
            torch.nn.Conv2d(
                n_input_channels, 
                conv_layer_sizes[0], 
                conv_kernel_sizes[0]))
        self.conv_network.append(
            torch.nn.MaxPool2d(2))
        
        # Rest of the Conv network
        for idx in range(len(self.conv_layer_sizes) - 1):
            self.conv_network.append(
                torch.nn.Conv2d(
                    self.conv_layer_sizes[idx], 
                    self.conv_layer_sizes[idx + 1], 
                    self.conv_kernel_sizes[idx + 1]))
            self.conv_network.append(
                torch.nn.Dropout(p=dropout))
            self.conv_network.append(
                torch.nn.MaxPool2d(2))

        # Dense network after Conv network
        self.dense_network = torch.nn.ModuleList()
        self.dense_network.append(
            torch.nn.Linear(
                self.calc_dense_n_inputs(),
                self.dense_layer_sizes[0]))
        
        # Rest of the Dense network
        for layer_size in self.dense_layer_sizes:
            self.dense_network.append(
                torch.nn.Linear(
                    layer_size,
                    layer_size
                )
            )
            
        self.output = torch.nn.Linear(self.dense_layer_sizes[0], 1)

    def calc_dense_n_inputs(self):
        """
        Assume no padding, 1 dilation, 1 stride in Conv2d layers.
        Assume no padding, 1 dilation, kernel size 2 in MaxPool2d layers.

        :returns: number of inputs into the dense (FC or Linear) network
        """      
        final_size = self.input_img_height
        for conv_kernel_size in self.conv_kernel_sizes:
            final_size = np.floor(final_size - (conv_kernel_size - 1))  # conv2d
            final_size = np.floor((final_size - (2-1) - 1)/2 + 1)  # maxpool2d
        return int(self.conv_layer_sizes[-1] * np.square(final_size))


    def forward(self, X, **kwargs):
        for layer in self.conv_network:
            if isinstance(layer, torch.nn.MaxPool2d):
                X = self.act_fn_maxpool(layer(X))
            else:
                X = layer(X)
        
        X = X.view(-1, X.size(1) * X.size(2) * X.size(3))
        
        for layer in self.dense_network:
            X = self.act_fn_dense(layer(X))
            
        return torch.nn.functional.softmax(X, dim=-1)
    
    
    
    
    
    
    
    
    
    
    
    