"""
Created on August 2022

@author: SimaoGoncalves99

Adversarial Training of a CNN for an unsupervised Domain Adaptation task. 
Inspired by the original work of Ganin et al. : "Domain-Adversarial Training of Neural Networks"
"""
#Inspired by:
#https://github.com/ciampluca/unsupervised_counting/blob/master/models/discriminator.py
#https://github.com/Yangyangii/DANN-pytorch/blob/master/DANN.ipynb

import torch.nn as nn
from torch.autograd import Function

#Gradiente Reversal Layer
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lamda):
        ctx.lamda=lamda
        
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        
        return grad_output.neg()*ctx.lamda,None
    

#Task Specific Classifier
class Classifier(nn.Module):
    """Classifier Network"""
    
    def __init__(self, input_size, num_classes):
        
        super(Classifier, self).__init__()
        

        self.layer = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(input_size, num_classes)
        )
        
       
        
    def forward(self, x):
        
        y = self.layer(x)
        
        return y


#Domain Discriminator
class Discriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
        #Inspired by:
        #https://github.com/Yangyangii/DANN-pytorch/blob/master/DANN.ipynb
    """
    def __init__(self, input_size, num_classes=1):
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, num_classes)
        )
        
    
    def forward(self, x, lamda):
        h = GradReverse.apply(x,lamda)
        y = self.layer(h)
        return y      

