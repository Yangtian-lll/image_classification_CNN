"""

QUESTION 1

Some helpful code for getting started.


"""

import torch
import torchvision
import torchvision.transforms as transforms
from imagenet10 import ImageNet10

import pandas as pd
import os

from config import *

# Gathers the meta data for the images
paths, classes = [], []
for i, dir_ in enumerate(CLASS_LABELS):
    for entry in os.scandir(ROOT_DIR + dir_):
        if (entry.is_file()):
            paths.append(entry.path)
            classes.append(i)

data = {
    'path': paths,
    'class': classes
}

data_df = pd.DataFrame(data, columns=['path', 'class'])
data_df = data_df.sample(frac=1).reset_index(drop=True)  # Shuffles the data

# See what the dataframe now contains
print("Found", len(data_df), "images.")
# If you want to see the image meta data
print(data_df.head())

# Split the data into train and test sets and instantiate our new ImageNet10 objects.




train_split = 0.80  # Defines the ratio of train/valid data.
train_split = 0.0072  # when use one batch , 64 images



# valid_size = 1.0 - train_size
train_size = int(len(data_df) * train_split)

data_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

dataset_train = ImageNet10(
    df=data_df[:train_size],
    transform=data_transform,
)

dataset_valid = ImageNet10(
    df=data_df[train_size:].reset_index(drop=True),
    transform=data_transform,
)

# Data loaders for use during training
train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

valid_loader = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

# See what you've loaded
print("len(dataset_train)", len(dataset_train))
print("len(dataset_valid)", len(dataset_valid))

print("len(train_loader)", len(train_loader))
print("len(valid_loader)", len(valid_loader))


# use one batch


import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# def a  model
class ConNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 4096),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # print("shape : " + str(x.shape) )
        x = x.view(x.size(0), 256 * 3 * 3)
        x = self.classifier(x)
        return x

class FullNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FullNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        x = x.reshape(-1, 128 * 128*3).to(device)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = ConNet(10).to(device)

model = FullNet(128*128*3, 1000,10).to(device)

learning_rate = 0.001

num_epochs = 40


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

def calval_loss(test_loader, model):

    model.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    test_loss = test_loss / len(test_loader.dataset)
    correct = correct.float() / len(test_loader.dataset)



    return test_loss , correct








# Train the model

train_loss_all = []
train_acc_all = []
val_loss_all = []
val_acc_all = []
epoch_done = []

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i + 1) % 100 == 0:
        #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
        #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    train_loss , train_acc =  calval_loss(train_loader, model)
    val_loss, val_acc = calval_loss(valid_loader, model)
    train_loss_all.append(train_loss)
    train_acc_all.append(train_acc)
    val_loss_all.append(val_loss)
    val_acc_all.append(val_acc)
    epoch_done.append(epoch)

    if (epoch + 1) % 1 == 0:
        torch.save(model.state_dict(), 'save/save_{}.pth'.format(epoch))

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (7,5))
plt.plot(epoch_done , train_loss_all)
plt.title('Train Loss')
plt.xlabel('epoch')
plt.ylabel('Train Loss')

fig1 = plt.figure(figsize = (7,5))
plt.plot(epoch_done ,train_acc_all)
plt.title('Train ACC')
plt.xlabel('epoch')
plt.ylabel('Train ACC')

fig2 = plt.figure(figsize = (7,5))
plt.plot(epoch_done , val_loss_all)
plt.title('Val Loss')
plt.xlabel('epoch')
plt.ylabel('Val Loss')

fig3 = plt.figure(figsize = (7,5))
plt.plot(epoch_done , val_acc_all)
plt.title('Val Acc')
plt.xlabel('epoch')
plt.ylabel('Val Acc')

plt.show()


# confus_matrix
import matplotlib.pyplot as plt
conf_matrix = torch.zeros(10, 10)
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

import  itertools
import numpy as np

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):



    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # 。。。。。。。。。。。。new code start。。。。。。。。。。。。。。。。
    # The x,y axes are the same length 
    plt.axis("equal")
    # Handle the x-axis if there are gaps on either side of the x- or y-axis 
    ax = plt.gca()  # Get current axis
    left, right = plt.xlim()  # Obtain the maximum and minimum values of the x-axis
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    # 。。。。。。。。。。。。new code end。。。。。。。。。。。。。。。。

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


conf_matrix = torch.zeros(10, 10)
for batch_images, batch_labels in valid_loader:
   # print(batch_labels)
   with torch.no_grad():
       if torch.cuda.is_available():
           batch_images, batch_labels = batch_images.cuda(),batch_labels.cuda()

   out = model(batch_images)

   prediction = torch.max(out, 1)[1]
   conf_matrix = confusion_matrix(prediction, labels=batch_labels, conf_matrix=conf_matrix)

CLASS_LABELS = [
  "baboon",
  "banana",
  "canoe",
  "cat",
  "desk",
  "drill",
  "dumbbell",
  "football",
  "mug",
  "orange",
]
plot_confusion_matrix(conf_matrix.numpy(), classes=CLASS_LABELS, normalize=False,
                                 title='Taining Set Normalized confusion matrix')

model.load_state_dict(torch.load(r'save/save_39.pth'))

test_dir = r'imagenet10/test_set'

imgs = os.listdir(test_dir)

print(imgs)

# pre_csv
from PIL import Image

name = []
result = []

for i in range(len(imgs)):

    img = Image.open(test_dir + '/' + imgs[i])

    img = img.convert('RGB')
    img = data_transform(img)
    input_batch = img.unsqueeze(0)
    input_batch = input_batch.cuda()
    outputs = model(input_batch)

    _, preds = outputs.max(1)

    result.append(preds.item())
    name.append(imgs[i])


data = {
    'image_name': name,
    'predicted_class_id': result
}

data_df = pd.DataFrame(data, columns=['image_name', 'predicted_class_id'])

data_df.to_csv('result.csv',index=False)





