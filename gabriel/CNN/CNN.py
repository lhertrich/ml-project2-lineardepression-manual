#import libraries

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        #convolutional layers - here 2 + pooling
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Input size
        self.fc2 = nn.Linear(128, num_classes)

        #l2 regularization - value of lambda
        self.l2_lambda = 0.001

    def forward(self, x, train=False):
        #1st Layer
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        #2nd Layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        #Flatten
        x = torch.flatten(x, start_dim=1)  #flattenn all dims expect batch

        #Fully connected layers
        x = F.relu(self.fc1(x))
        if train:
            x = F.dropout(x, p=0.3, training=train)  # Dropout - only do that during training - Adjust probability p
        x = self.fc2(x)

        return x

#Initialize the model
num_classes = num_classes
model = CNNModel(num_classes=num_classes)

#Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)  #parameters to be modified

#Example batch size and data dimensions
batch_size = 64
train_size = 60000 

#Learning rate decay schedule
def lr_scheduler(optimizer, epoch, base_lr=0.01, decay_rate=0.95, decay_epoch=1):
    if epoch % decay_epoch == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate
    return optimizer

#Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(DataLoader(...)):  # Replace with your dataset
        #Move data to GPU if available
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
            model = model.cuda()

        #Forward pass
        outputs = model(images, train=True)
        loss = criterion(outputs, labels)

        #Add L2 regularization
        l2_reg = sum(torch.norm(param) for param in model.parameters())
        loss += model.l2_lambda * l2_reg

        #Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    #Update learning rate
    optimizer = lr_scheduler(optimizer, epoch)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

#Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in DataLoader(...):  # Replace with your dataset
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")



