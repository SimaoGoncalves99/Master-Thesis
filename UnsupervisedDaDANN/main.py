"""
Created on August 2022

@author: SimaoGoncalves99


Adversarial Training of a CNN for an unsupervised Domain Adaptation task. 
Inspired by the original work of Ganin et al. : "Domain-Adversarial Training of Neural Networks"

This is the main file of the project. This task consists of a melanoma vs benign classification task.

"""

import torchvision
import torch
import torch.nn as nn
from torchvision import transforms
import utils as utils
import Networks as net
from train import Trainer
import pickle

#Clear memory
torch.cuda.empty_cache()

# Configure Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


#Transformations to apply to the datasets
transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
     ])

transform_val = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
     ])


#Load data
training_data_S = torchvision.datasets.ImageFolder(root = "./source/training",transform=transform_train)
validation_data_S = torchvision.datasets.ImageFolder(root = "./source/validation",transform=transform_val)

training_data_T = torchvision.datasets.ImageFolder(root = "./target/training",transform=transform_train) 
validation_data_T = torchvision.datasets.ImageFolder(root = "./target/validation",transform=transform_val) 


# Create data loaders for our datasets; shuffle for training
training_loader_S = torch.utils.data.DataLoader(training_data_S, batch_size=32, shuffle=True)
validation_loader_S = torch.utils.data.DataLoader(validation_data_S, batch_size=32, shuffle=False)
training_loader_T = torch.utils.data.DataLoader(training_data_T, batch_size=32, shuffle=True)
validation_loader_T = torch.utils.data.DataLoader(validation_data_T, batch_size=32, shuffle=False)

# Class labels
classes = ('benign', 'melanoma')

# Report split sizes
print('Source Training set has {} instances'.format(len(training_data_S)))
print('Target Training set has {} instances'.format(len(training_data_T)))
print('Source Validation set has {} instances'.format(len(validation_data_S)))
print('Target Validation set has {} instances'.format(len(validation_data_T)))

#Create the feature extractor, the classifier and the discriminator  (Considering the ResNet18 architecture)
feature_extractor = torchvision.models.resnet18(pretrained=True)
num_features = feature_extractor.fc.in_features
feature_extractor.fc = nn.Identity()


classifier = net.Classifier(input_size=num_features,num_classes = len(classes))

discriminator = net.Discriminator(input_size=num_features)

#Pass models to the GPU
F = utils.to_cuda(feature_extractor) 
C = utils.to_cuda(classifier)
D = utils.to_cuda(discriminator)


print(F)
print(C)
print(D)

#Hyperparameters (Tunable!)
lr = 5e-7
batch_size = 32
num_epochs = 100
step = 0
n_critic = 1 # for training more k steps about Discriminator
n_batches = len(training_data_S)//batch_size


#Loss function and optimizer (C)
loss_C = torch.nn.CrossEntropyLoss(weight = torch.Tensor([0.02,0.98])).to(device)
optimizer_C = torch.optim.Adam(C.parameters(),lr=lr,betas=(0.9,0.999),weight_decay=0)

#Loss function and optimizer (D)
loss_D =  nn.BCEWithLogitsLoss().to(device)
optimizer_D = torch.optim.Adam(params=D.parameters(), lr=lr, betas=(0.9, 0.999),weight_decay = 0) 

#Optimizer (F)
optimizer_F = torch.optim.Adam(params=F.parameters(), lr=lr, betas=(0.9, 0.999),weight_decay = 0) 


#Train the siamese network model using Adam optimizer
trainer_adam = Trainer( 
    F=F,
    D=D,
    C=C,
    dataloader_train_S=training_loader_S,
    dataloader_train_T = training_loader_T,
    dataloader_test_S = validation_loader_S,
    dataloader_test_T = validation_loader_T,
    loss_C = loss_C,
    loss_D = loss_D,
    optimizer_C = optimizer_C,
    optimizer_F = optimizer_F,
    optimizer_D = optimizer_D  
)

trainer_adam_value = True
track_val = 0

track_train_loss_total,track_train_C_loss,track_train_D_loss,track_train_acc,track_train_acc_domain,track_val_loss,track_val_acc, recall_list,specificity_list,precision_list,balanced_acc_list,f1_list,confusion_list,fpr_list,tpr_list,threshold_list,auc_resnet18_list,track_val_loss_target,track_val_acc_target,recall_list_target,specificity_list_target,precision_list_target,balanced_acc_list_target,f1_list_target,confusion_list_target,fpr_list_target,tpr_list_target,threshold_list_target,auc_resnet18_list_target,epoch = trainer_adam.train(num_epochs, batch_size, n_batches, step, device)

#Store the data 
#(General)
pickle.dump(track_train_loss_total,open("track_train_loss_total.dat","wb"))
pickle.dump(track_train_C_loss,open("track_train_C_loss.dat","wb"))
pickle.dump(track_train_D_loss,open("track_train_D_loss.dat","wb"))
pickle.dump(track_train_acc,open("track_train_acc.dat","wb"))
pickle.dump(track_train_acc_domain,open("track_train_acc_domain.dat","wb"))

#(Source)
pickle.dump(track_val_loss,open("track_val_loss.dat","wb"))
pickle.dump(track_val_acc,open("track_val_acc.dat","wb"))
pickle.dump(f1_list,open("f1_list.dat","wb"))
pickle.dump(confusion_list,open("confusion_list.dat","wb"))
pickle.dump(precision_list,open("precision_list.dat","wb"))
pickle.dump(balanced_acc_list,open("balanced_acc_list.dat","wb"))
pickle.dump(recall_list,open("recall_list.dat","wb"))
pickle.dump(specificity_list,open("specificity_list.dat","wb"))
pickle.dump(fpr_list,open("fpr_list.dat","wb"))
pickle.dump(tpr_list,open("tpr_list.dat","wb"))
pickle.dump(threshold_list,open("threshold_list.dat","wb"))
pickle.dump(auc_resnet18_list,open("auc_resnet18_list.dat","wb"))

#(Target)
pickle.dump(track_val_loss_target,open("track_val_loss_target.dat","wb"))
pickle.dump(track_val_acc_target,open("track_val_acc_target.dat","wb"))
pickle.dump(f1_list_target,open("f1_list_target.dat","wb"))
pickle.dump(confusion_list_target,open("confusion_list_target.dat","wb"))
pickle.dump(precision_list_target,open("precision_list_target.dat","wb"))
pickle.dump(balanced_acc_list_target,open("balanced_acc_list_target.dat","wb"))
pickle.dump(recall_list_target,open("recall_list_target.dat","wb"))
pickle.dump(specificity_list_target,open("specificity_list_target.dat","wb"))
pickle.dump(fpr_list_target,open("fpr_list_target.dat","wb"))
pickle.dump(tpr_list_target,open("tpr_list_target.dat","wb"))
pickle.dump(threshold_list_target,open("threshold_list_target.dat","wb"))
pickle.dump(auc_resnet18_list_target,open("auc_resnet18_list_target.dat","wb"))


#Final print
print(f"Final Total Training Loss: {track_train_loss_total[epoch]}. Final Training Classification Loss: {track_train_C_loss[epoch]}. Final Training Discriminator Loss: {track_train_D_loss[epoch]}. Final training Accuracy: {track_train_acc[epoch]}. Final training Accuracy Domain: {track_train_acc_domain[epoch]}\n")
   
print(f"Final Validation Loss (Source): {track_val_loss[epoch]}. Final Validation Accuracy (Source): {track_val_acc[epoch]}\n")
print(f"F1 score (Source): {f1_list[epoch]}.\n Confusion-Matrix (Source): {confusion_list[epoch]}.\n Precision (Source): {precision_list[epoch]}.\n Balanced-accuracy (Source): {balanced_acc_list[epoch]}.\n Recall (Source): {recall_list[epoch]}.\n Specificity (Source): {specificity_list[epoch]}.\n AUC (Source): {auc_resnet18_list[epoch]}.\n")
    
print(f"Final Validation Loss (Target): {track_val_loss_target[epoch]}. Final Validation Accuracy (Target): {track_val_acc_target[epoch]}\n")
print(f"F1 score (Target): {f1_list_target[epoch]}.\n Confusion-Matrix (Target): {confusion_list_target[epoch]}.\n Precision (Target): {precision_list_target[epoch]}.\n Balanced-accuracy (Target): {balanced_acc_list_target[epoch]}.\n Recall (Target): {recall_list_target[epoch]}.\n Specificity (Target): {specificity_list_target[epoch]}.\n AUC (Target): {auc_resnet18_list_target[epoch]}.\n")
