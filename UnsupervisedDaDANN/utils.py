"""
Created on August 2022

@author: SimaoGoncalves99

Adversarial Training of a CNN for an unsupervised Domain Adaptation task. 
Inspired by the original work of Ganin et al. : "Domain-Adversarial Training of Neural Networks"

"""

import torch
import numpy as np
import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc
import pickle

def validate_source(dataloader, F, C, D, loss_C, optimizer_F, optimizer_C, epoch, best_model_metric):
    """
    Performs validation on the source data
    """
    F.eval()
    C.eval()
    D.eval()
    
    # Tracking variables
    loss_avg = 0
    total_correct = 0
    total_images = 0
    total_steps = 0
    prediction_list = []
    ground_truth_list = []
    
    with torch.no_grad():  # No need to compute gradient when testing
       
        for (X_batch, Y_batch) in dataloader:
            
            # Pass images to the gpu
            X_batch, Y_batch = to_cuda([X_batch, Y_batch])
            
            #Extract features from the source images
            source_features = F(X_batch)
            
            #Evaluate the extracted features
            source_logits = C(source_features)   
            
            # Compute the classification loss
            loss = loss_C(source_logits, Y_batch)

            # Predicted class is the max index over the column dimension
            predictions = source_logits.argmax(dim=1).squeeze()
            Y_batch = Y_batch.squeeze()

            # Update tracking variables
            loss_avg += loss.cpu().item()
            total_steps += 1
            
            total_correct += (predictions == Y_batch).sum().cpu().item()
            total_images += predictions.shape[0]
            
            prediction_list.append(predictions.cpu().detach().numpy()) 
            ground_truth_list.append(Y_batch.cpu().detach().numpy())        
            
    #Track validation loss and accuracy
    loss_avg = loss_avg / total_steps
    accuracy = total_correct / total_images
    
    #Concatenate the prediction lists
    prediction_list = np.concatenate(prediction_list, axis=0)
    ground_truth_list = np.concatenate(ground_truth_list, axis=0 )
    
    #Evaluate metrics
    f1 = f1_score(ground_truth_list,prediction_list)
    confusion = confusion_matrix(ground_truth_list,prediction_list)
    precision = precision_score(ground_truth_list,prediction_list)
    balanced_acc = balanced_accuracy_score(ground_truth_list,prediction_list)
    recall = recall_score(ground_truth_list,prediction_list)
    specificity = (confusion[0,0])/(confusion[0,0]+confusion[0,1])
    fpr,tpr,threshold = roc_curve(ground_truth_list,prediction_list)  #fpr = false positive rate, tpr = true positive rate
    auc_model= auc(fpr,tpr)

    #Check if the model and the metrics should be saved
    if best_model_metric < auc_model:
     
      #Update best metrics
      best_model_metric = auc_model
      best_epoch = epoch
      best_f1 = f1
      best_confusion = confusion 
      best_precision = precision
      best_balanced_acc = balanced_acc
      best_recall = recall
      best_specificity = specificity
      best_fpr = fpr
      best_tpr = tpr
      best_threshold = threshold
      best_auc = auc_model

      #Store the data 
      pickle.dump(best_model_metric,open("best_model_metric.dat","wb"))
      pickle.dump(best_epoch,open("best_epoch.dat","wb"))
      pickle.dump(best_f1,open("best_f1.dat","wb"))
      pickle.dump(best_confusion,open("best_confusion.dat","wb"))
      pickle.dump(best_precision,open("best_precision.dat","wb"))
      pickle.dump(best_balanced_acc,open("best_balanced_acc.dat","wb"))
      pickle.dump(best_recall,open("best_recall.dat","wb"))
      pickle.dump(best_specificity,open("best_specificity.dat","wb"))
      pickle.dump(best_fpr,open("best_fpr.dat","wb"))
      pickle.dump(best_tpr,open("best_tpr.dat","wb"))
      pickle.dump(best_threshold,open("best_threshold.dat","wb"))
      pickle.dump(best_auc,open("best_auc.dat","wb"))

      #Save the ft extractor checkpoint
      checkpoint_F = {
          "epoch":best_epoch,
          "model_state":F.state_dict()
      }
      torch.save(checkpoint_F,"best_model_ft_extractor_DA.pth")
      
      #Save the classifier checkpoint
      checkpoint_C = {
          "epoch":best_epoch,
          "model_state":C.state_dict()
      }
      torch.save(checkpoint_C,"best_model_classifier_DA.pth")
      
      #Save the discriminator checkpoint
      checkpoint_D = {
          "epoch":best_epoch,
          "model_state":D.state_dict()
      }
      torch.save(checkpoint_D,"best_model_discriminator_DA.pth")
    
    #Always save the models in every epoch, might be useful later
    
    #Save the ft extractor checkpoint
    checkpoint_F_epoch = {
        "epoch":epoch,
        "model_state":F.state_dict()
    }
    torch.save(checkpoint_F_epoch,f"models_per_epoch/extractors/model_ft_extractor_DA_{epoch}.pth")
    
    #Save the classifier checkpoint
    checkpoint_C_epoch = {
        "epoch":epoch,
        "model_state":C.state_dict()
    }
    torch.save(checkpoint_C_epoch,f"models_per_epoch/class_classifiers/model_classifier_DA_{epoch}.pth")
    
    #Save the discriminator checkpoint
    checkpoint_D_epoch = {
        "epoch":epoch,
        "model_state":D.state_dict()
    }
    torch.save(checkpoint_D_epoch,f"models_per_epoch/domain_classifiers/model_discriminator_DA_{epoch}.pth")
    
    
    print(f"Validation loss(Source): {loss_avg}. Validation accuracy(Source): {accuracy}\n")
    print(f"F1 score(Source): {f1}.\n Confusion-Matrix(Source): {confusion}.\n Precision(Source): {precision}.\n Balanced-accuracy(Source): {balanced_acc}.\n Recall(Source): {recall}.\n Specificity(Source): {specificity}.\n AUC(Source): {auc_model}.\n")
    print("\n")
    
    return best_model_metric, prediction_list, ground_truth_list, loss_avg, accuracy, f1, confusion, precision, balanced_acc, recall, specificity, fpr, tpr, threshold, auc_model

def validate_target(dataloader, F, C, D, loss_C, optimizer_F, optimizer_C, epoch, best_model_metric):
    """
    Computes the total loss and accuracy over the whole dataloader
    Args:
        dataloder: Test dataloader
        model: torch.nn.Module
        loss_function: The loss criterion, e.g: nn.CrossEntropyLoss()
    Returns:
        [loss_avg, accuracy]: both scalar.
    """
    F.eval()
    C.eval()
    D.eval()

    # Tracking variables
    loss_avg = 0
    total_correct = 0
    total_images = 0
    total_steps = 0
    prediction_list = []
    ground_truth_list = []
    
    with torch.no_grad():  # No need to compute gradient when testing
       
        for (X_batch, Y_batch) in dataloader:
            
            # Pass images to the gpu
            X_batch, Y_batch = to_cuda([X_batch, Y_batch])
            
            
            #Extract features from the source images
            target_features = F(X_batch)
            
            #Evaluate the extracted features
            target_logits = C(target_features)   
            
            # Compute the classification loss
            loss = loss_C(target_logits, Y_batch)

            # Predicted class is the max index over the column dimension
            predictions = target_logits.argmax(dim=1).squeeze()
            Y_batch = Y_batch.squeeze()

            # Update tracking variables
            loss_avg += loss.cpu().item()
            total_steps += 1
            
            total_correct += (predictions == Y_batch).sum().cpu().item()
            total_images += predictions.shape[0]
            
            prediction_list.append(predictions.cpu().detach().numpy()) 
            ground_truth_list.append(Y_batch.cpu().detach().numpy())        
            
    #Track validation loss and accuracy
    loss_avg = loss_avg / total_steps
    accuracy = total_correct / total_images
    
    #Concatenate the prediction lists
    prediction_list = np.concatenate(prediction_list, axis=0)
    ground_truth_list = np.concatenate(ground_truth_list, axis=0 )
    
    #Evaluate metrics
    f1 = f1_score(ground_truth_list,prediction_list)
    confusion = confusion_matrix(ground_truth_list,prediction_list)
    precision = precision_score(ground_truth_list,prediction_list)
    balanced_acc = balanced_accuracy_score(ground_truth_list,prediction_list)
    recall = recall_score(ground_truth_list,prediction_list)
    specificity = (confusion[0,0])/(confusion[0,0]+confusion[0,1])
    fpr,tpr,threshold = roc_curve(ground_truth_list,prediction_list)  #fpr = false positive rate, tpr = true positive rate
    auc_model= auc(fpr,tpr)

    #Check if the model and the metrics should be saved
    if best_model_metric < auc_model:
     
      #Update best metrics
      best_model_metric = auc_model
      best_epoch = epoch
      best_f1 = f1
      best_confusion = confusion 
      best_precision = precision
      best_balanced_acc = balanced_acc
      best_recall = recall
      best_specificity = specificity
      best_fpr = fpr
      best_tpr = tpr
      best_threshold = threshold
      best_auc = auc_model

      #Store the data 
      pickle.dump(best_model_metric,open("best_model_metric_target.dat","wb"))
      pickle.dump(best_epoch,open("best_epoch_target.dat","wb"))
      pickle.dump(best_f1,open("best_f1_target.dat","wb"))
      pickle.dump(best_confusion,open("best_confusion_target.dat","wb"))
      pickle.dump(best_precision,open("best_precision_target.dat","wb"))
      pickle.dump(best_balanced_acc,open("best_balanced_acc_target.dat","wb"))
      pickle.dump(best_recall,open("best_recall_target.dat","wb"))
      pickle.dump(best_specificity,open("best_specificity_target.dat","wb"))
      pickle.dump(best_fpr,open("best_fpr_target.dat","wb"))
      pickle.dump(best_tpr,open("best_tpr_target.dat","wb"))
      pickle.dump(best_threshold,open("best_threshold_target.dat","wb"))
      pickle.dump(best_auc,open("best_auc_target.dat","wb"))

      #Save the ft extractor checkpoint
      checkpoint_F = {
          "epoch":best_epoch,
          "model_state":F.state_dict()
      }
      torch.save(checkpoint_F,"best_model_ft_extractor_DA_target.pth")
      
      #Save the classifier checkpoint
      checkpoint_C = {
          "epoch":best_epoch,
          "model_state":C.state_dict()
      }
      torch.save(checkpoint_C,"best_model_classifier_DA_target.pth")
      
      #Save the discriminator checkpoint
      checkpoint_D = {
          "epoch":best_epoch,
          "model_state":D.state_dict()
      }
      torch.save(checkpoint_D,"best_model_discriminator_DA_target.pth")
      
    
    print(f"Validation loss(Target): {loss_avg}. Validation accuracy(Target): {accuracy}\n")
    print(f"F1 score(Target): {f1}.\n Confusion-Matrix(Target): {confusion}.\n Precision(Target): {precision}.\n Balanced-accuracy(Target): {balanced_acc}.\n Recall(Target): {recall}.\n Specificity(Target): {specificity}.\n AUC(Target): {auc_model}.\n")
    print("\n")
    
    return best_model_metric, prediction_list, ground_truth_list, loss_avg, accuracy, f1, confusion, precision, balanced_acc, recall, specificity, fpr, tpr, threshold, auc_model



def to_cuda(elements):
    """
    Transfers all parameters/tensors to GPU memory (cuda) if there is a GPU available
    """
    if not torch.cuda.is_available():
        return elements
    if isinstance(elements, tuple) or isinstance(elements, list):
        return [x.cuda() for x in elements]
    return elements.cuda()