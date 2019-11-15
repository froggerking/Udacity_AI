# Toni Boehm, Training a CNN- Achitecture with a directory and save the trained model.

import matplotlib.pyplot as plt
import numpy as np
import time
from numba import cuda
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
from matplotlib.ticker import FormatStrFormatter
from collections import OrderedDict
import sys,os
import math
from matplotlib.ticker import FormatStrFormatter
import argparse

#hier kommen die ARGPARSE Abfragen hin:


#Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=False, help="trains on GPU", action="store_true")
parser.add_argument("--path", default="/home/workspace/ImageClassifier/flowers", help="path to training data")
parser.add_argument("--structure", default= "vgg16", help="CNN structure")
parser.add_argument("--learning_rate", default=0.002, type=float, help="learning rate")
parser.add_argument("--hidden_units", default=5000, type=int, help="hidden units of classifier")
parser.add_argument("--dropout", default=0.8, type=float, help="Dropout between 0 and 1")
parser.add_argument("--epochs", default=3, type= int, help="Number of Epochs")
parser.add_argument("--save_dir", default="/home/workspace/ImageClassifier/checkpoint.pth", help="path and filename to save Model")
args = parser.parse_args()

#Set variables to parser:
if args.gpu == True:
    device_set = "cuda"
else:
    device_set = "cpu"
data_dir = args.path
structure = args.structure
learning_rate = args.learning_rate
hidden_units = args.hidden_units
dropout = args.dropout
epochs_set = args.epochs
save_path = args.save_dir

print (device_set)
print (data_dir)
print (structure)
print (learning_rate)
print (hidden_units)
print (dropout)
print (epochs_set)
print (save_path)

print("Input Directory is set to:", data_dir)

train_dir = data_dir + "/train" 
valid_dir = data_dir + "/valid"
test_dir = data_dir + "/test"


# Define transforms for the training, validation
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])


# Load the train and validation datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)

# Use the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
vloader = torch.utils.data.DataLoader(validation_data, batch_size =64,shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)

print ("Classes in Trainloader:", len(trainloader))
print("Number of Pictures:", len(trainloader.dataset))


# Building and training the classifier
#Structures for different CNNs
structures = {"vgg16":25088,
              "densenet121" : 1024,
              "alexnet" : 9216 }

#Setup of Network classifier

def nn_setup(structure='vgg16',dropout=0.9, hidden_units = 5000, learning_rate = 0.002):
    
    # Modell zuweisen, wird keines übergeben wird vgg16 verwendet. Bei falscher Eingabe wird Alexnet verwendet.
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)        
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print ("Modell {} nicht bekannt, es wird automatisch alexnet verwendet".format (structure))
        model = models.alexnet(pretrained = True)
        structure="alexnet"
        
        
    # Training für das Modell aussetzen    
    for param in model.parameters():
        param.requires_grad = False

    # Definition des Classifiers
    model.classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(dropout)),
            ('inputs', nn.Linear(structures[structure], hidden_units)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_units, 200)),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(200,len(trainloader))),
            ('output', nn.LogSoftmax(dim=1)) ]))
        
        
    # model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    #model.cuda()
        
    return model , optimizer ,criterion 

    

model,optimizer,criterion = nn_setup(structure,dropout,hidden_units,learning_rate)


# Train the CNN on training data


def training (epochs = 6, print_every = 15, device = device_set):
    #starting points:
    steps = 0
    loss_show=[]
    t0 = time.time()

    print ("Modell wir nach {} transferiert".format(device))
    # change to cuda
    model.to(device)
    print ("Starting Training mit Epochen:", epochs)
    for e in range(epochs):
        running_loss = 0
        for i, (inputs, labels) in enumerate(trainloader): #for ii, (inputs, labels) in enumerate(trainloader):
            timeset = time.time()
            steps += 1
            
            if device == "cuda":
                inputs,labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy=0
                
                
                for i, (inputs2, labels2) in enumerate(vloader): 
                    optimizer.zero_grad()
                    inputs2, labels2 = inputs2.to(device) , labels2.to(device)
                    model.to(device)
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        vlost = criterion(outputs,labels2).item() 
                        ps = torch.exp(outputs).data 
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                        
                vlost = vlost / len(vloader)
                accuracy = accuracy /len(vloader)
                
                        
                
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost {:.4f}".format(vlost),
                       "Accuracy: {:.4f}".format(accuracy),
                     "Time: {:.4f}".format (time.time()-timeset))
                
                
                running_loss = 0
    
                
    t1 = time.time()
    
    total = t1-t0
    print ("Trainingtime was", time.time()-t0)
        
    

training(epochs_set,10,device_set)


# Testing your network

def testing (testloader,device_test="cpu"):
    correct = 0
    total = 0
    model.to(device_test)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device_test), labels.to(device_test)
            outputs = model(images)
            loose, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("'Accuracy of the trained model: ", (100 * correct / total), "%")
print ("start testing the model")    
testing (testloader, device_set)


# Save model and all important data to checkpoint

model.class_to_idx = train_data.class_to_idx
model.cpu
torch.save({'structure' :'vgg16',
            'hidden_units':hidden_units,
            'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx},
            save_path)

print ("model is trained and saved to:", save_path)
