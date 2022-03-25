import enum
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader

from PIL import Image
import torch.nn.functional as F

path = '/Users/raymondgreenfield/Documents/Classes/Spring 2022/Deep Learning/kaggle/input/homework4'

bird_class_df = pd.read_csv('/Users/raymondgreenfield/Documents/Classes/Spring 2022/Deep Learning/kaggle/input/homework4/birds_400/birds.csv') 

# Hyper Parameters 
input_size = 4320000         # this is 224x224 I think this is mainly for logistic regression
num_classes = 400          # Length of bird in csv file
num_epochs = 10            # this is fine may need to increase it 
batch_size = 1           # tune batch size according to CPU or the GPU
learning_rate = 0.001

# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        out = self.linear(x)
        return out

def import_data():
    if os.path.exists(path):
            for dirname, _, filenames in os.walk(path):    
                for filename in filenames:
                    #print(os.path.join(dirname, filename))
                    os.path.join(dirname, filename)
    else:
        print("Error: Path does not exist")

#CNN 
class ConvNet(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(ConvNet, self).__init__()

        # Output size after convolution filter
        #(w-f+2p)/s + 1 

        # Input shape = (256, 3, 150, 150) batchsize, RGB, image dimension
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=2, padding=1),
        # Shape= (256,12,150,150)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Shape= (256,12,150,150)
        self.relu1 = nn.ReLU()
        # Shape= (256,12,150,150) (nonlinearity)

        self.pool = nn.MaxPool2d(kernel_size=2) 
        # Reduce the image size by a factor of 2
        # Shape= (256,12,75,75)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=2, padding=1),
        # Shape= (256,20,150,150)
        self.relu2 = nn.ReLU()
        # Shape= (256,12,75,75)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=2, padding=1),
        # Shape= (256,32,75,75)
        self.bn3 = nn.BatchNorm2d(num_features=12)
        # Shape= (256,32,75,75)
        self.relu3 = nn.ReLU()
        # Shape= (256,32,75,75)

        # Adding the fully connected layer
        self.fc = nn.Linear(in_features=32*75*75, out_features=num_classes) # this could be tested on the LogisticRegression

        # Feed Forward function

    def forward(self, inputs):

        output = self.conv1(inputs)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        # Above output will be in matrix form, with shape (256,32,75,75)

        output = output.view(-1,32*75*75)
        output = self.fc(output)

        return output

def create_test_train_valid_df():
    # I do not know why my general filepath is not working in MacOS
    directories = ['/Users/raymondgreenfield/Documents/Classes/Spring 2022/Deep Learning/kaggle/input/homework4/birds_400/test',
                   '/Users/raymondgreenfield/Documents/Classes/Spring 2022/Deep Learning/kaggle/input/homework4/birds_400/train',
                   '/Users/raymondgreenfield/Documents/Classes/Spring 2022/Deep Learning/kaggle/input/homework4/birds_400/valid']

    for dir in directories:
        label = []
        path = []
        for dirname, _,filenames in os.walk(dir):
            for filename in filenames:
                label.append(os.path.split(dirname)[1])
                path.append(os.path.join(dirname,filename))
        if dir == directories[0]:
            df_test = pd.DataFrame(columns=['path','label'])
            df_test['path']=path
            df_test['label']=label
        elif dir == directories[1]:
            df_train = pd.DataFrame(columns=['path','label'])
            df_train['path']=path
            df_train['label']=label        
        elif dir == directories[2]:
            df_valid = pd.DataFrame(columns=['path','label'])
            df_valid['path']=path
            df_valid['label']=label

    return df_test, df_train, df_valid

def plot_traing_df(df_train):
    # Display 20 picture of the dataset with their labels
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 7),
                            subplot_kw={'xticks': [], 'yticks': []})

    df_sample = df_train.sample(10)
    df_sample.reset_index(drop=True, inplace=True)

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(df_sample.path[i]))
        ax.set_title(df_sample.label[i])
    plt.tight_layout()
    plt.show()

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize = (12,12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[: 100], 10).permute(1,2,0))
        break

def main():
    import_data()
    """
    df_test, df_train, df_valid = create_test_train_valid_df()   
    #(2001, 2), (58388, 2), (2000, 2) respectively
    #plot_traing_df(df_train)

    df_sample = df_train.sample(10)
    df_sample.reset_index(drop=True, inplace=True)

    #print(df_sample.path[0])

    image = Image.open(df_sample.path[0])

    # Define a transform to convert the image to tensor
    transform = transforms.ToTensor()

    # Convert the image to PyTorch tensor
    tensor = transform(image)

    # print the shape of converted image tensor
    #print(tensor.shape) 
    #torch.Size([3, 224, 224])

    #plt.imshow(transforms.ToPILImage()(transforms.ToTensor()(image)), interpolation="bicubic")
    #plt.show()
    # 
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"                     
    kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
    

    trans = transforms.Compose([                                           # Defining a variable transforms
                        transforms.Resize((150,150)),                      # Resize the image to 256x256 pixels
                        #transforms.CenterCrop(224),                       # Crop the image to 224x224 pixels about the center
                        transforms.ToTensor(),                             # 0-255 to 0-1 numpy to tensor
                        transforms.Normalize(mean=(0.5, 0.5, 0.5),         # Normalize the image
                                            std= (0.5, 0.5, 0.5))           # Mean and std of image as also used when training
                        ])
    
    # PyTorch DataLoader
    train_path = '/Users/raymondgreenfield/Documents/Classes/Spring 2022/Deep Learning/kaggle/input/homework4/birds_400/train'
    test_path = '/Users/raymondgreenfield/Documents/Classes/Spring 2022/Deep Learning/kaggle/input/homework4/birds_400/test'
    valid_path = '/Users/raymondgreenfield/Documents/Classes/Spring 2022/Deep Learning/kaggle/input/homework4/birds_400/valid'

    train_loader = DataLoader(ImageFolder(train_path, transform=trans), 
                              batch_size=64, shuffle=True, #num_workers=3, pin_memory=True
                            )
    test_loader  = DataLoader(ImageFolder(test_path, transform=trans), 
                              batch_size=32, shuffle=True, #num_workers=3, pin_memory=True
                            )
    valid_loader = DataLoader(ImageFolder(valid_path, transform=trans), 
                              batch_size, #num_workers=3, pin_memory=True
                            )
    
    #dataiter = iter(train_loader)
    #data = dataiter.next()
    #print(data)

    

    root = pathlib.Path(train_path)
    classes = sorted(j.name.split('/')[-1] for j in root.iterdir())             
    
    model = ConvNet(num_classes).to(device)

    # Optimizer and loss function
    # Parameters have been set as global values
    loss_function = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)  

    train_count = len(glob.glob(train_path+'/**/*.jpg'))
    test_count = len(glob.glob(test_path+'/**/*.jpg'))

    # Model training and saving best model
    best_accuracy = 0.0


    for epoch in range(num_epochs):

        # Evaluation and training on training dataset
        model.train()                                   #Tells model we are in the training phase
        train_accuracy = 0.0
        train_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():               # I dont know if I should do this or not 
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images.cpu())
                labels = Variable(labels.cpu())            

            optimizer.zero_grad()                       # We do not want to mix up the gradients between mini-batches at the start of a new batch

            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward() 
            loss.step()

            train_loss += loss.cpu().data*images.size(0)
            _, prediction = torch.max(outputs.data,1)

            train_accuracy += int(torch.sum(prediction==labels.data))

        train_accuracy = train_accuracy/train_count
        train_loss = train_loss/train_count


        # Evaluation on testing dataset
        model.eval()

        test_accuracy = 0.0 
        for i, (images, labels) in enumerate(test_loader):
            if torch.cuda.is_available():               # I dont know if I should do this or not 
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images.cpu())
                labels = Variable(labels.cpu())


            outputs = model(images)
            _, prediction = torch.max(outputs.data,1)
            test_accuracy += int(torch.sum(prediction==labels.data))

        test_accuracy = test_accuracy/test_count

        print('Epoch: ' + str(epoch) + 'Train Loss: ' + str(int(train_loss)) + 'Train Accuracy: ' + str(train_accuracy) + 'Test Accuracy: ' + str(test_accuracy))

        if test_accuracy > best_accuracy:
            torch.save(model.state_dict(), 'best_checkpoint.model')
            best_accuracy = test_accuracy
    

    """    
    model = LogisticRegression(input_size, num_classes).to(device)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

    # Training the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 4320000))       #This has to change
            labels = Variable(labels)
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, num_epochs, i+1, loss.data))
    
    """
if __name__ == '__main__':
    main()

