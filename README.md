# NN_Pytorch

### Task 1 :  Neural Networks Models for Binary Classifcation 5 Data Sets

### Data Description

five binary data sets 

### Settings of the neural network models

This neural network has one hidden layer with Relu and output layer with Softmax.
The number of units in the hidden layer will be tuned from 1 to 10 for all five datasets, respectively.

For optimizer, Stochastic gradient descent (SGD) is an iterative optimization algorithm used in the training model. Simply, optimization is a searching process for finding the global optimal for minimizing loss function by updating parameters in models.

By using k-fold, a validation set, and a training set are made to feed NN to train. 

After setting the number of input features for each dataset, the models having a different number of units for hidden layers from 1 to 10 trained on validation, training, and test dataset. The updating parameter is also conducted during the training model with cross-validation using cross-entropy.


### Task 2  :  Neural Networks Models for Multi-class Data Sets

### Data Description

-  Multi-class Data Sets: containing a training set of 10,000 examples and a test set of 1,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. 


### Settings of the neural network models

Neural Network has one more layer in between the input layer and the output layer, having two hidden layers which denoted as L1 and L2 respectively. The number of input units is fixed as 784, and the number of units of L1 and L2 will be tuned.

The learning rate is set up as 0.003 with Epoch = 30 and Batch Size = 15. 
 
The training dataset is split into training and validation datasets by using the sklearn test and train split method. 

There are the specific number of units for each hidden later, which are 50, 75, 100 for L1 and 10, 15 , 20 L2. By using cross- validation with cross-entropy, the number of the units will be tuned
 

 
 
 
 






