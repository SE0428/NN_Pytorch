import os
import numpy as np
import pandas as pd
import torch

torch.manual_seed(0)  # for reproducibility
import torch.nn as nn
import time
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics


os.chdir('./datasets/bi-class')


def average(lst):
    return round(sum(lst) / len(lst), 3)


def label(nn_output):
    # nn_output is softmax value from nn that has two units : tensor
    # print(nn_output.data)
    labels = []

    for i in range(len(nn_output)):

        if nn_output[i][0] > nn_output[i][1]:
            labels.append(0)
        else:
            labels.append(1)

    # print(len(labels))
    return labels


class BinaryClassifcation(nn.Module):

    def __init__(self, num):
        super(BinaryClassifcation, self).__init__()
        self.num_features = num
        self.unit = 1  # for initial

        # Number of input features is 10.
        # Inputs to 1st hidden layer linear transformation
        self.hidden = nn.Linear(self.num_features,
                                self.unit)  # number of hidden unit need to be decided from cross validation

        # Output layer,1 units
        self.output = nn.Linear(self.unit, 2)

        # Define sigmoid activation and softmax output
        self.relu = nn.ReLU()  #
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        # x->hidden->softmax->output (probability of each class )
        # or classify 0 or 1 : softmax

        # output of first layer
        x = self.hidden(x)
        x = self.relu(x)

        # output
        x = self.output(x)
        x = self.softmax(x)

        return x


def train(data):  # data from nzp file
    # ready k-fold for cross validation
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    # for data frame
    num_unit = range(1, 11)
    average_loss = []  # average loss on k-fold dataset
    acc = []  # accuracy on test data
    avg_time = []  # computaion time
    auc = []

    # confusion matrix
    tn_ = []
    fp_ = []
    fn_ = []
    tp_ = []

    for j in range(1, 11):

        loss_list = []
        time_list = []

        # create NN for binary classifcation
        num_features = len(data['train_X'][0])
        model = BinaryClassifcation(num_features)
        # change the number of units in hidden layer
        model.unit = j  # j 1 to 10

        # loss Function
        criterion = torch.nn.CrossEntropyLoss()

        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)

        # print("number of hidden units:",model.unit)

        for train_index, test_index in kf.split(data['train_X']):
            start_time = time.time()

            # k-fold cross validation
            X_train, X_test = torch.FloatTensor(data['train_X'][train_index]), torch.FloatTensor(
                data['train_X'][test_index])
            y_train, y_test = torch.LongTensor(data['train_Y'][train_index]), torch.LongTensor(
                data['train_Y'][test_index])
            # print(train_index)
            # print(X_train)

            output = model(X_train)
            # print(output)
            loss = criterion(output, y_train)
            loss_list.append(loss.item())

            loss.backward()
            optimizer.step()

            time_list.append(time.time() - start_time)

        # print("Average loss on K-fold:",average(loss_breast))
        average_loss.append(average(loss_list))

        # train model for test data to get accuracy of the model with
        test_X, test_Y = torch.FloatTensor(data['test_X']), torch.LongTensor(data['test_Y'])
        nn_output = model(test_X)
        test_result = label(nn_output)

        # accuracy_score(y_true, y_pred)
        accuracy = accuracy_score(test_Y, test_result)

        # train area under curve (auc)
        fpr, tpr, thresholds = metrics.roc_curve(test_Y, test_result, pos_label=1)
        auc.append(metrics.auc(fpr, tpr))  # false positive true positive

        tn, fp, fn, tp = confusion_matrix(test_Y, test_result).ravel()

        tn_.append(tn)
        fp_.append(fp)
        fn_.append(fn)
        tp_.append(tp)

        # print("Accuracy on Test Data:",round(accuracy,3))
        acc.append(round(accuracy, 3))
        avg_time.append(average(time_list))

    df = pd.DataFrame(zip(num_unit, average_loss, acc, auc, avg_time, tn_, fp_, fn_, tp_),
                      columns=['# of units', 'Avg_loss', "acc_test", "auc", "Avg_time", "tn", "fp", "fn", "tp"])

    print(df)

    return df



##main##

#five data set
breast_cancer=np.load('./breast-cancer.npz')
diabets=np.load('./diabetes.npz')
digit=np.load('./digit.npz')
iris=np.load('./iris.npz')
wine=np.load('./wine.npz')



#five data set training
train(breast_cancer)
train(diabets)
train(digit)
train(iris)
train(wine)
