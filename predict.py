# Image Classifier orediction modeul. Loads a trained model, inputs a picture and predicts the classes based on a json dictionary

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
import pwd
import math
import getpass
import json
from matplotlib.ticker import FormatStrFormatter
import argparse


#Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--image", default="/home/workspace/ImageClassifier/flowers/test/1/image_06743.jpg", help="path to Image")
parser.add_argument("--checkpoint", default= "/home/workspace/ImageClassifier/checkpoint.pth", help="Model to be used")
parser.add_argument("--gpu", default=False, help="uses GPU/CPU ", action="store_true")
parser.add_argument("--top_k", default=5, help="How many Top_K")
parser.add_argument("--category_names", default= "/home/workspace/ImageClassifier/cat_to_name.json", help="Cat to name file (json)")
args = parser.parse_args()

#Set variables to parser:
if args.gpu == True:
    device_set = "cuda"
else:
    device_set = "cpu"

test_image = args.image
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names

#give Feedback to settings:
print ("Will use {} as the Device".format (device_set))
print ("Will use {} as the image to classify".format (test_image))
print ("Will use modell from {}".format (checkpoint))
print ("Will show {} most likely classes".format (top_k))
print ("Will use the {} dictionary for labeling".format (category_names))

# load Cat to Name Dictionary
#category_names = "/home/workspace/aipnd-project/cat_to_name.json"
print("category Directory is set to:", category_names)
with open(category_names, "r") as f:
    cat_to_name = json.load(f)


structures = {"vgg16":25088,
              "densenet121" : 1024,
              "alexnet" : 9216 }

# Define Structure
def nn_setup(structure='vgg16',dropout=0.6, hidden_units = 5000, learning_rate = 0.001):
    
    # Modell zuweisen, wird keines übergeben wird vgg16 verwendet. Bei falscher Eingabe wird Alexnet verwendet.
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)        
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print ("Modell {} nicht bekannt, es wird automatisch alexnet verwendet".format (structure))
        model = models.alexnet(pretrained = True)
        structure="alexnet"
        
        
    # No Training for the model    
    for param in model.parameters():
        param.requires_grad = False

    # Definition des Classifiers
    model.classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(dropout)),
            ('inputs', nn.Linear(structures[structure], hidden_units)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_units, 200)),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(200,103)),
            ('output', nn.LogSoftmax(dim=1)) ]))
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    return model , optimizer ,criterion 

#load model from checkpoint:
def load_model(path):
    checkpoint = torch.load('checkpoint.pth')
    structure = checkpoint['structure']
    hidden_units = checkpoint['hidden_units']
    model,_,_ = nn_setup(structure, 0.6,hidden_units)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    pil_image = Image.open(image)
    im_resized = pil_image.resize((224,224))
    np_image = np.array(im_resized)
    np_image = np_image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose(2, 0, 1)
    return np_image


def predict(image_path, model, top_k, device_set):   
    model.to(device_set)
    img_pip = process_image(image_path)
    img_tensor = torch.from_numpy(img_pip)
    img_tensor = img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.float()
    with torch.no_grad():
        output = model.forward(img_tensor.to(device_set))
    probability = F.softmax(output.data,dim=1)
    return probability.topk(top_k)
  
model = load_model(checkpoint)
probs, classes = predict (test_image, model, top_k, device_set)
probs = probs.data.numpy().squeeze()
classes = classes.data.numpy().squeeze()+1

for i in range(0, len(classes)):
    print('The most likely class is: {}; probability: {}'.format(cat_to_name[str(classes[i])], probs[i]))
