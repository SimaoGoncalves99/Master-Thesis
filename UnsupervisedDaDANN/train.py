"""
Created on August 2022

@author: SimaoGoncalves99

Adversarial Training of a CNN for an unsupervised Domain Adaptation task. 
Inspired by the original work of Ganin et al. : "Domain-Adversarial Training of Neural Networks"

"""

import torch
import sys
import tqdm
import collections
from torchvision import datasets, transforms
import utils as utils
import numpy as np
import pickle


def get_lambda(epoch, num_epochs):
    #epoch+=1
    p = epoch / (num_epochs-1)
    return 2. / (1+np.exp(-10.*p)) - 1.

class Trainer:

    def __init__(self, F, D, C, dataloader_train_S, dataloader_train_T, dataloader_test_S, dataloader_test_T, loss_C,loss_D,optimizer_C, optimizer_F,optimizer_D):
       
        self.dataloader_train_S = dataloader_train_S
        self.dataloader_train_T = dataloader_train_T
        self.dataloader_test_S = dataloader_test_S
        self.dataloader_test_T = dataloader_test_T
        self.F = F
        self.C = C
        self.D = D
        self.loss_C = loss_C
        self.loss_D = loss_D
        self.optimizer_C = optimizer_C
        self.optimizer_F = optimizer_F
        self.optimizer_D = optimizer_D
        
        
    def train(self, num_epochs, batch_size, num_batches, step, device):
        
        
        track_train_loss = []
       
        track_train_C_loss = []
        track_train_D_loss = []
        track_train_loss_total = []
        
        #Track the average lambda value per epoch
        track_avg_lambda = []
       
        track_train_acc = []
        track_train_acc_domain = []
        track_val_loss = []
        track_val_acc = []
        f1_list = []
        confusion_list = []
        precision_list = []
        balanced_acc_list = []
        recall_list = []
        specificity_list = []
        fpr_list = []
        tpr_list = []
        threshold_list = []
        auc_model_list = []
        best_model_metric = float('-inf') 
       
        track_val_loss_target = []
        track_val_acc_target = []
        f1_list_target = []
        confusion_list_target = []
        precision_list_target = []
        balanced_acc_list_target = []
        recall_list_target = []
        specificity_list_target = []
        fpr_list_target = []
        tpr_list_target = []
        threshold_list_target = []
        auc_model_list_target = []
        best_model_metric_target = float('-inf')
        
        
        for epoch in range(num_epochs):
            
            avg_C_loss = avg_D_loss = avg_loss_total = avg_lambda = avg_lr = running_correct_pred = running_correct_pred_domain = total_images = total_images_domain = 0
            
            C_losses_train = []
            D_losses_train = []
            Total_losses_train = []

            total_steps = 0
            
            self.F.train()
            self.D.train()
            self.C.train()
            
            #Iterate through the source data
            for i, (source_images,source_labels) in enumerate(tqdm.tqdm(self.dataloader_train_S,desc=f" Adversarial Training epoch {epoch}")): 
                
                #Sample target images
                target_images, target_labels = next(iter(self.dataloader_train_T)) 
                
                # Defining labels for adversarial training
                D_src = torch.ones(source_images.shape[0], 1).to(device) # Discriminator Label to source
                D_tgt = torch.zeros(target_images.shape[0], 1).to(device) # Discriminator Label to target
                D_labels = torch.cat([D_src, D_tgt], dim=0)

                #Pass images and labels to cuda
                source_images,source_labels = source_images.to(device), source_labels.to(device)
                target_images = target_images.to(device)
                
                # ##########
                # #TRAINING#
                # ##########
                
                # Setting grads to zero
                self.optimizer_D.zero_grad()
                self.optimizer_F.zero_grad()
                self.optimizer_C.zero_grad()
                
                #Concatenate source and target images and pass them through the Feature Extractor
                x = torch.cat([source_images,target_images], dim=0)
                h = self.F(x)
                
                #Extract the source features
                source_features = h[:source_images.shape[0]]
                
                #Compute the lambda value
                lambda_C_D = get_lambda(epoch, num_epochs)
                
                #Pass the features extracted from both datasets through the Domain Discriminator
                y = self.D(h,lambda_C_D)    
                
                #Pass the source features through the classifier
                source_logits = self.C(source_features)

                #Compute the classification loss and the discriminator loss
                loss_C = self.loss_C(source_logits, source_labels)
                loss_D = self.loss_D(y, D_labels)
               
                #Compute the total loss
                loss_total = loss_C + loss_D
                
                #Perform backpropagation
                loss_total.backward()
                
                #Update all the parameters
                self.optimizer_D.step()
                self.optimizer_C.step()
                self.optimizer_F.step()
                
        
            ##############    
            #COMPUTATIONS#
            ##############    
            
                # Compute the classification loss, the discriminator loss and the total loss
                C_losses_train.append(loss_C.cpu().detach().item()) #Save the classification losses per batch 
                D_losses_train.append(loss_D.cpu().detach().item()) #Save the classification losses per batch 
                Total_losses_train.append(loss_total.cpu().detach().item()) #Save the classification losses per batch 
               
                # Average losses
                avg_C_loss += loss_C.cpu().detach().item()
                avg_D_loss += loss_D.cpu().detach().item()
                avg_loss_total += loss_total.cpu().detach().item()
                
                # Average lambda
                avg_lambda += lambda_C_D

                
                total_steps+=1
                
                #Training predictions (class classifier)
                predictions = source_logits.argmax(dim=1).squeeze()
                source_labels = source_labels.squeeze()
                
                #Track training accuracy in each epoch
                running_correct_pred += (predictions == source_labels).sum().cpu().item()
 
                total_images += source_images.shape[0] 
                print(predictions.shape)
                
                #Training predictions (domain classifier)
                predictions_domain = torch.sigmoid(y).squeeze()
                predictions_domain = torch.round(predictions_domain)
                D_labels = D_labels.squeeze()
                
                #Track training accuracy in each epoch
                running_correct_pred_domain += (predictions_domain == D_labels).sum().cpu().item()
                total_images_domain += predictions_domain.shape[0]
               
            ##############    
            #COMPUTATIONS#
            ##############       
               
            #Compute the average training loss per epoch
            avg_C_loss = (avg_C_loss)/(total_steps)
            avg_D_loss = (avg_D_loss)/(total_steps)
            avg_loss_total = (avg_loss_total)/(total_steps)
            
            #Compute the average lambda per epoch
            avg_lambda = (avg_lambda)/(total_steps)
            
            #Compute the average lr per epoch
            avg_lr = (avg_lr)/total_steps
            
            #Track the training loss per epoch
            track_train_C_loss.append(avg_C_loss)
            track_train_D_loss.append(avg_D_loss)
            track_train_loss_total.append(avg_loss_total)
            
            #Track the average lambda value per epoch
            track_avg_lambda.append(avg_lambda)
            
            
            avg_C_loss = avg_D_loss = avg_loss_total = avg_lambda = total_steps = 0
            
            #Track the training accuracy per epoch
            train_acc = running_correct_pred/total_images
            track_train_acc.append(train_acc)
            train_acc = 0
            
            #Track the domain classification accuracy per epoch
            train_acc_domain = running_correct_pred_domain/total_images_domain
            track_train_acc_domain.append(train_acc_domain)
            train_acc_domain = 0
            
            #Print training loss and accuracy
            print(f'Training Total Loss: {track_train_loss_total[epoch]}')
            print(f'Training Classification Loss: {track_train_C_loss[epoch]}')
            print(f'Training Discriminator Loss: {track_train_D_loss[epoch]}')
            print(f'Average lambda value: {track_avg_lambda[epoch]}')
            print(f'Training Accuracy: {track_train_acc[epoch]}')
            print(f'Training Accuracy Domain: {track_train_acc_domain[epoch]}\n')
            
            #Perform validation
            
            best_model_metric, prediction_list, ground_truth_list,val_loss, val_acc, f1, confusion, precision, balanced_acc, recall,specificity, fpr, tpr, threshold, auc_model = utils.validate_source(self.dataloader_test_S, self.F, self.C, self.D, self.loss_C, self.optimizer_F, self.optimizer_C, epoch, best_model_metric)
            best_model_metric_target, prediction_list_target, ground_truth_list_target,val_loss_target, val_acc_target, f1_target, confusion_target, precision_target, balanced_acc_target, recall_target,specificity_target, fpr_target, tpr_target, threshold_target, auc_model_target = utils.validate_target(self.dataloader_test_T, self.F, self.C, self.D, self.loss_C, self.optimizer_F, self.optimizer_C, epoch, best_model_metric_target)   
            
                
            #Save source results
            track_val_loss.append(val_loss)
            track_val_acc.append(val_acc)
            f1_list.append(f1)
            confusion_list.append(confusion)
            precision_list.append(precision)
            balanced_acc_list.append(balanced_acc)
            recall_list.append(recall)
            specificity_list.append(specificity)
            auc_model_list.append(auc_model)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            threshold_list.append(threshold)
            
            #Save target results
            track_val_loss_target.append(val_loss_target)
            track_val_acc_target.append(val_acc_target)
            f1_list_target.append(f1_target)
            confusion_list_target.append(confusion_target)
            precision_list_target.append(precision_target)
            balanced_acc_list_target.append(balanced_acc_target)
            recall_list_target.append(recall_target)
            specificity_list_target.append(specificity_target)
            auc_model_list_target.append(auc_model_target)
            fpr_list_target.append(fpr_target)
            tpr_list_target.append(tpr_target)
            threshold_list_target.append(threshold_target)
            
            

        return track_train_loss_total,track_train_C_loss,track_train_D_loss,track_train_acc,track_train_acc_domain,track_val_loss,track_val_acc, recall_list,specificity_list,precision_list,balanced_acc_list,f1_list,confusion_list,fpr_list,tpr_list,threshold_list,auc_model_list,track_val_loss_target,track_val_acc_target,recall_list_target,specificity_list_target,precision_list_target,balanced_acc_list_target,f1_list_target,confusion_list_target,fpr_list_target,tpr_list_target,threshold_list_target,auc_model_list_target,epoch
