"""
Created on August 2022

@author: SimaoGoncalves99


Training of a CNN for an unsupervised Domain Adaptation task. 
Inspired by the original work of Sun et al. : "Deep CORAL: Correlation Alignment for Deep Domain Adaptation"

This is the main file of the project. This task consists of a melanoma vs benign classification task.
The source and target datasets are not provided and should be placed on the folders "./source/training", 
"./source/validation", "./target/training", and "./target/validation".
"""

import torchvision
import torch
import torch.nn as nn
from torchvision import transforms
import coral_utils as utils
from coral_train import Trainer
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

#Create the model (RESNET 18)
feature_extractor = torchvision.models.resnet18(pretrained=True)
num_features = feature_extractor.fc.in_features
    
classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features,len(classes))
    )

feature_extractor.fc = nn.Identity()

#Pass extractor and classifier to the GPU
feature_extractor = utils.to_cuda(feature_extractor)
classifier = utils.to_cuda(classifier)

#Print the models
print(feature_extractor)
print(classifier)

#Hyperparameters
lr = 5e-7
num_epochs =  100 
lambda_coral = 100 #Hyperparameter that defines the trade-off between the classification task loss and the domain adaptation loss. (Tunable)

#Loss function (Use the source data weight distribution)
loss_function = torch.nn.CrossEntropyLoss(weight = torch.Tensor([0.02,0.98])).to(device)


optimizer_F = torch.optim.Adam(feature_extractor.parameters(),lr,betas=(0.9,0.999),weight_decay=0)
optimizer_C = torch.optim.Adam(classifier.parameters(),lr,betas=(0.9,0.999),weight_decay=0)

#Train model (Source Data Controls Loop)
trainer_adam = Trainer(
  feature_extractor = feature_extractor,
  classifier = classifier,
  dataloader_train_S = training_loader_S,
  dataloader_train_T = training_loader_T,
  dataloader_test_S = validation_loader_S,
  dataloader_test_T = validation_loader_T,
  batch_size = 32,
  loss_function = loss_function,
  optimizer_F = optimizer_F,
  optimizer_C = optimizer_C
)

#Train the model (Source Controls Loop)
h,source_features,target_features,track_train_loss_total,track_train_C_loss,track_train_D_loss,track_train_acc,track_val_loss,track_val_acc, recall_list,specificity_list,precision_list,balanced_acc_list,f1_list,confusion_list,fpr_list,tpr_list,threshold_list,auc_resnet18_list,track_val_loss_target,track_val_acc_target,recall_list_target,specificity_list_target,precision_list_target,balanced_acc_list_target,f1_list_target,confusion_list_target,fpr_list_target,tpr_list_target,threshold_list_target,auc_resnet18_list_target,epoch = trainer_adam.train(num_epochs,lambda_coral, device)

#Store the data 
#(General)
pickle.dump(track_train_loss_total,open("track_train_loss_total.dat","wb"))
pickle.dump(track_train_C_loss,open("track_train_C_loss.dat","wb"))
pickle.dump(track_train_D_loss,open("track_train_D_loss.dat","wb"))
pickle.dump(track_train_acc,open("track_train_acc.dat","wb"))

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


#Final print (Source Controls Loop)

print(f"Final Total Training Loss: {track_train_loss_total[epoch]}. Final Training Classification Loss: {track_train_C_loss[epoch]}. Final Training CORAL Loss: {track_train_D_loss[epoch]}. Final training Accuracy: {track_train_acc[epoch]}.\n")
   
print(f"Final Validation Loss (Source): {track_val_loss[epoch]}. Final Validation Accuracy (Source): {track_val_acc[epoch]}\n")
print(f"F1 score (Source): {f1_list[epoch]}.\n Confusion-Matrix (Source): {confusion_list[epoch]}.\n Precision (Source): {precision_list[epoch]}.\n Balanced-accuracy (Source): {balanced_acc_list[epoch]}.\n Recall (Source): {recall_list[epoch]}.\n Specificity (Source): {specificity_list[epoch]}.\n AUC (Source): {auc_resnet18_list[epoch]}.\n")
    
print(f"Final Validation Loss (Target): {track_val_loss_target[epoch]}. Final Validation Accuracy (Target): {track_val_acc_target[epoch]}\n")
print(f"F1 score (Target): {f1_list_target[epoch]}.\n Confusion-Matrix (Target): {confusion_list_target[epoch]}.\n Precision (Target): {precision_list_target[epoch]}.\n Balanced-accuracy (Target): {balanced_acc_list_target[epoch]}.\n Recall (Target): {recall_list_target[epoch]}.\n Specificity (Target): {specificity_list_target[epoch]}.\n AUC (Target): {auc_resnet18_list_target[epoch]}.\n")