from Multiclass_dataload import fileload
from scipy.io import loadmat
import pandas as pd
import sys
import numpy as np
import pandas as pd
import seaborn as sns

#from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(0) #for reproducibility

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# for test data
test_images = loadmat('./test_images.mat')
test_labels = loadmat('./test_labels.mat')

# for training data
train_images = loadmat('./train_images.mat')
train_labels = loadmat('./train_labels.mat')

# file load from Multiclass_dataload
#train = fileload('train',train_images,train_labels)
#test = fileload('test',test_images,test_labels)

train = pd.read_csv("train.csv", index_col=[0])
test = pd.read_csv("test.csv", index_col=[0])


def accuracy(y_pred, y_test):
    # y_pred_softmax = torch.log_softmax(y_pred, dim = 1)

    _, y_pred_tags = torch.max(y_pred, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc) * 100

    return acc


def label(output):
    _, predicted_label = torch.max(output, dim=1)
    return predicted_label


class Setdata(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class MultiClass(nn.Module):

    def __init__(self, num_L1_unit, num_L2_unit):
        super(MultiClass, self).__init__()
        self.num_features = 784
        self.num_L1_units = num_L1_unit
        self.num_L2_units = num_L2_unit

        # Number of input features is 10.
        # Inputs to 1st hidden layer linear transformation
        self.L1 = nn.Linear(self.num_features,
                            self.num_L1_units)  # number of hidden unit need to be decided from cross validation
        self.L2 = nn.Linear(self.num_L1_units, self.num_L2_units)

        # Output layer,10 units
        self.output = nn.Linear(self.num_L2_units, 10)

        # Define sigmoid activation and softmax output
        self.relu = nn.ReLU()  #
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        #

        x = self.L1(x)
        x = self.relu(x)

        x = self.L2(x)
        x = self.relu(x)

        x = self.output(x)
        x = self.softmax(x)

        return x

# train(valid),test
train_X = train.iloc[:, 0:-1]
train_y = train.iloc[:, -1]

# randomly sampling 80% of the training instances to train a classifier and then testing it on the remaining 20%
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.2, random_state=42)
X_test = test.iloc[:, 0:-1]
y_test = test.iloc[:, -1]

# data noramlize
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)

train_dataset = Setdata(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
val_dataset = Setdata(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
test_dataset = Setdata(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

# Setting parameters
EPOCHS = 30
BATCH_SIZE = 15
LEARNING_RATE = 0.003
NUM_FEATURES = len(test.columns) - 1  # exclude label
NUM_CLASSES = 10

# data loader -> iterable over a dataset
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)

# check if GPU is avaliable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


#for graph
acc_list = {'train_50_10': [], "val_50_10": [], 'train_50_15': [], "val_50_15": [], \
            'train_50_20': [], "val_50_20": [], 'train_75_10': [], "val_75_10": [], \
            'train_75_15': [], "val_75_15": [], 'train_75_20': [], "val_75_20": [],\
            'train_100_10': [], "val_100_10": [], 'train_100_15': [], "val_100_15": [], \
            'train_100_20': [], "val_100_20": []}

loss_list = {'train_50_10': [], "val_50_10": [], 'train_50_15': [], "val_50_15": [], \
             'train_50_20': [], "val_50_20": [], 'train_75_10': [], "val_75_10": [], \
             'train_75_15': [], "val_75_15": [], 'train_75_20': [], "val_75_20": [],\
             'train_100_10': [], "val_100_10": [], 'train_100_15': [], "val_100_15": [], \
             'train_100_20': [], "val_100_20": []}

test_predicted_list = {

    '50_10': [], '50_15': [], '50_20': [], \
    '75_10': [], '75_15': [], '75_20': [], \
    '100_10': [], '100_15': [], '100_20': [],
}

# number of each layer's units
L1 = [50, 75, 100] #number of first layer units
L2 = [10, 15, 20] #number of second layer units

#training part
for num_L1 in L1:

    for num_L2 in L2:
        print("First units:", num_L1, "Second units:", num_L2, "begin training.")

        model = MultiClass(num_L1, num_L2)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        #for e in tqdm(range(1, EPOCHS + 1)): #jupyter notebook
        for e in range(1, EPOCHS + 1):
            train_epoch_loss = 0
            train_epoch_acc = 0

            model.train()

            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)

                optimizer.zero_grad()

                train_output = model(X_train_batch)

                train_loss = criterion(train_output, y_train_batch)
                train_acc = accuracy(train_output, y_train_batch)

                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()

                # validation
            with torch.no_grad():  # spped up the computation

                val_epoch_loss = 0
                val_epoch_acc = 0

                model.eval()
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                    val_output = model(X_val_batch)

                    val_loss = criterion(val_output, y_val_batch)
                    val_acc = accuracy(val_output, y_val_batch)

                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()

            ind1 = 'train_' + str(num_L1) + "_" + str(num_L2)
            ind2 = 'val_' + str(num_L1) + "_" + str(num_L2)

            loss_list[ind1].append(train_epoch_loss / len(train_loader))
            loss_list[ind2].append(val_epoch_loss / len(val_loader))
            acc_list[ind1].append(train_epoch_acc / len(train_loader))
            acc_list[ind2].append(val_epoch_acc / len(val_loader))

            print(f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | \
            Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| \
            Val Acc: {val_epoch_acc / len(val_loader):.3f}')

        # get result of each model on test data
        with torch.no_grad():

            print("train on test data L1:", num_L1, "L2:", num_L2)

            model.eval()

            #print(model.num_L1_units)
            #print(model.num_L2_units)

            ind = str(num_L1) + "_" + str(num_L2)

            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)

                test_output = model(X_batch)
                predicted_label = label(test_output)

                test_predicted_list[ind].append(predicted_label.cpu().numpy())

        test_predicted_list[ind] = [a.squeeze().tolist() for a in test_predicted_list[ind]]


# Create dataframes
train_val_acc_df = pd.DataFrame.from_dict(acc_list).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_list).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})

# Plot the dataframes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
plt.show()


confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, test_predicted_list['50_20']))

sns.heatmap(confusion_matrix_df, annot=True)
plt.show()


confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, test_predicted_list['100_20']))
sns.heatmap(confusion_matrix_df, annot=True)
plt.show()
