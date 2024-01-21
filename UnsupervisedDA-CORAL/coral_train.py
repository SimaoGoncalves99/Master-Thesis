"""
Created on August 2022

@author: SimaoGoncalves99


Training of a CNN for an unsupervised Domain Adaptation task. 
Inspired by the original work of Sun et al. : "Deep CORAL: Correlation Alignment for Deep Domain Adaptation"

"""

import torch
import tqdm
import coral_utils as utils

class Trainer:

    def __init__(self, feature_extractor,classifier, dataloader_train_S, dataloader_train_T, dataloader_test_S, dataloader_test_T,batch_size, loss_function , optimizer_F, optimizer_C):
       
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.dataloader_train_S = dataloader_train_S
        self.dataloader_train_T = dataloader_train_T
        self.dataloader_test_S = dataloader_test_S
        self.dataloader_test_T = dataloader_test_T
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.optimizer_F = optimizer_F
        self.optimizer_C = optimizer_C
        
        
    def train(self, num_epochs, lambda_coral, device):
       
        track_train_C_loss = []
        track_train_D_loss = []
        track_train_loss_total = []

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
        auc_resnet18_list = []
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
        auc_resnet18_list_target = []
        best_model_metric_target = float('-inf')
        
        
        for epoch in range(num_epochs):
            
            avg_C_loss = avg_D_loss = avg_loss_total = running_correct_pred = total_images = 0
            
            C_losses_train = []
            D_losses_train = []
            Total_losses_train = []

            total_steps = 0
            
            self.feature_extractor.train()
            self.classifier.train()

            
            #Iterate through the in vivo data
            for i, (source_images,source_labels) in enumerate(tqdm.tqdm(self.dataloader_train_S,desc=f" DA (CORAL) Training epoch {epoch}")): 
                
                #Sample target images
                target_images, target_labels = next(iter(self.dataloader_train_T)) 

                #Pass images and labels to cuda
                source_images,source_labels = source_images.to(device), source_labels.to(device)
                target_images = target_images.to(device)
                
                # ##########
                # #TRAINING#
                # ##########
                
                # Setting grads to zero
                self.optimizer_F.zero_grad()
                self.optimizer_C.zero_grad()

                #Concatenate source and target images and pass them through the feature extractor
                x = torch.cat([source_images,target_images], dim=0)
                h = self.feature_extractor(x)
                
                #Extract the source and the target features
                source_features = h[:source_images.shape[0]]
                target_features = h[source_images.shape[0]:]
                
                
                #Print the feature dimensions
                #print(source_features.size())
                #print(target_features.size())
                
                #Pass the source features through the classifier
                source_logits = self.classifier(source_features)
                
                
                #Compute the CORAL loss (FEATURE LEVEL)
                loss_D = utils.coral(source_features, target_features)
               

                #Compute the classification loss
                loss_C = self.loss_function(source_logits, source_labels)
               
                #Compute the total loss
                loss_total = loss_C + lambda_coral*loss_D 
                
                #Perform backpropagation
                loss_total.backward()
                
                #Update all the parameters
                self.optimizer_F.step()
                self.optimizer_C.step()
        
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
                

                total_steps+=1
                
                #Training predictions (classifier)
                predictions = source_logits.argmax(dim=1).squeeze()
                source_labels = source_labels.squeeze()
                
                #Track training accuracy in each epoch
                running_correct_pred += (predictions == source_labels).sum().cpu().item()
                total_images += source_images.shape[0]#predictions.shape[0] #EfficientNet problems not sure why 
                
               
            ##############    
            #COMPUTATIONS#
            ##############       
               
            #Compute the average training loss per epoch
            avg_C_loss = (avg_C_loss)/(total_steps)
            avg_D_loss = (avg_D_loss)/(total_steps)
            avg_loss_total = (avg_loss_total)/(total_steps)
            
            
            #Track the training loss per epoch
            track_train_C_loss.append(avg_C_loss)
            track_train_D_loss.append(avg_D_loss)
            track_train_loss_total.append(avg_loss_total)
            
            avg_C_loss = avg_D_loss = avg_loss_total = total_steps = 0
            
            #Track the training accuracy per epoch
            train_acc = running_correct_pred/total_images
            track_train_acc.append(train_acc)
            train_acc = 0
            
            #Print training loss and accuracy
            print(f'Training Total Loss: {track_train_loss_total[epoch]}')
            print(f'Training Classification Loss: {track_train_C_loss[epoch]}')
            print(f'Training CORAL Loss: {track_train_D_loss[epoch]}')
            print(f'Training Accuracy: {track_train_acc[epoch]}')

            
            #Perform validation
            
            best_model_metric, prediction_list, ground_truth_list,val_loss, val_acc, f1, confusion, precision, balanced_acc, recall,specificity, fpr, tpr, threshold, auc_resnet18 = utils.validate_source(self.dataloader_test_S, self.feature_extractor, self.classifier, self.loss_function, self.optimizer_F, self.optimizer_C, epoch, best_model_metric)
            best_model_metric_target, prediction_list_target, ground_truth_list_target,val_loss_target, val_acc_target, f1_target, confusion_target, precision_target, balanced_acc_target, recall_target,specificity_target, fpr_target, tpr_target, threshold_target, auc_resnet18_target = utils.validate_target(self.dataloader_test_T, self.feature_extractor, self.classifier, self.loss_function, self.optimizer_F, self.optimizer_C, epoch, best_model_metric_target)   
            
                
            #Save source results
            track_val_loss.append(val_loss)
            track_val_acc.append(val_acc)
            f1_list.append(f1)
            confusion_list.append(confusion)
            precision_list.append(precision)
            balanced_acc_list.append(balanced_acc)
            recall_list.append(recall)
            specificity_list.append(specificity)
            auc_resnet18_list.append(auc_resnet18)
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
            auc_resnet18_list_target.append(auc_resnet18_target)
            fpr_list_target.append(fpr_target)
            tpr_list_target.append(tpr_target)
            threshold_list_target.append(threshold_target)
            
            
           
        return h,source_features,target_features,track_train_loss_total,track_train_C_loss,track_train_D_loss,track_train_acc,track_val_loss,track_val_acc, recall_list,specificity_list,precision_list,balanced_acc_list,f1_list,confusion_list,fpr_list,tpr_list,threshold_list,auc_resnet18_list,track_val_loss_target,track_val_acc_target,recall_list_target,specificity_list_target,precision_list_target,balanced_acc_list_target,f1_list_target,confusion_list_target,fpr_list_target,tpr_list_target,threshold_list_target,auc_resnet18_list_target,epoch
