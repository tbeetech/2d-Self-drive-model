# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 19:12:48 2022

@author: Oluwatobi
"""
#modules required 
#to save and load the exprience replay memory on the system
import os
#to sample random transitions from the experience replay memory. 
import random0
#import pytorch libary 
import torch
#to construct nueral network architecture
import torch.nn as nn
#to foward propagate neural network input(weight, bias)  
import torch.nn.functional as F
#to optimize the gradient of the loss function with respect network input parameters
import torch.optim as optim

from torch.autograd import Variable



class Network(nn.module):
    def __init__(self, input_size, nb_action):
        super().__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
        
    def foward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values
             
# class ReplayMemory(Object):
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#     def push(self, event):
#         self.memory.append(event)
#         if len(self.memory) > self.capacity:
#             del self.memory[0]
#         def sample(self, batch_size):
       
         
            