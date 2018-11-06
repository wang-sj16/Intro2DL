
# coding: utf-8

# # Homework-2: MLP for MNIST Classification
# 
# ### **Deadline: 2018.11.04 23:59:59**
# 
# ### In this homework, you need to
# - #### implement SGD optimizer (`./optimizer.py`)
# - #### implement forward and backward for FCLayer (`layers/fc_layer.py`)
# - #### implement forward and backward for SigmoidLayer (`layers/sigmoid_layer.py`)
# - #### implement forward and backward for ReLULayer (`layers/relu_layer.py`)
# - #### implement EuclideanLossLayer (`criterion/euclidean_loss.py`)
# - #### implement SoftmaxCrossEntropyLossLayer (`criterion/softmax_cross_entropy.py`)

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf

from network import Network
from solver import train, test
from plot import plot_loss_and_acc


# ## Load MNIST Dataset
# We use tensorflow tools to load dataset for convenience.

# In[2]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# In[3]:


def decode_image(image):
    # Normalize from [0, 255.] to [0., 1.0], and then subtract by the mean value
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [784])
    image = image / 255.0
    image = image - tf.reduce_mean(image)
    return image

def decode_label(label):
    # Encode label with one-hot encoding
    return tf.one_hot(label, depth=10)


# In[4]:


# Data Preprocessing
x_train = tf.data.Dataset.from_tensor_slices(x_train).map(decode_image)
y_train = tf.data.Dataset.from_tensor_slices(y_train).map(decode_label)
data_train = tf.data.Dataset.zip((x_train, y_train))

x_test = tf.data.Dataset.from_tensor_slices(x_test).map(decode_image)
y_test = tf.data.Dataset.from_tensor_slices(y_test).map(decode_label)
data_test = tf.data.Dataset.zip((x_test, y_test))


# ## Set Hyerparameters
# You can modify hyerparameters by yourself.

# In[5]:


batch_size = 100
max_epoch = 20
init_std = 0.01

learning_rate_SGD = 0.001
weight_decay = 0.5

disp_freq = 50


# ## 1. MLP with Euclidean Loss
# In part-1, you need to train a MLP with **Euclidean Loss**.  
# **Sigmoid Activation Function** and **ReLU Activation Function** will be used respectively.
# ### TODO
# Before executing the following code, you should complete **./optimizer.py** and **criterion/euclidean_loss.py**.

# In[6]:


from criterion import EuclideanLossLayer
from optimizer import SGD

criterion = EuclideanLossLayer()

sgd = SGD(learning_rate_SGD, weight_decay)


# ## 1.1 MLP with Euclidean Loss and Sigmoid Activation Function
# Build and train a MLP contraining one hidden layer with 128 units using Sigmoid activation function and Euclidean loss function.
# 
# ### TODO
# Before executing the following code, you should complete **layers/fc_layer.py** and **layers/sigmoid_layer.py**.

# In[7]:


from layers import FCLayer, SigmoidLayer

sigmoidMLP = Network()
# Build MLP with FCLayer and SigmoidLayer
# 128 is the number of hidden units, you can change by your own
sigmoidMLP.add(FCLayer(784, 128))
sigmoidMLP.add(SigmoidLayer())
sigmoidMLP.add(FCLayer(128, 10))


# In[15]:


sigmoidMLP, sigmoid_loss, sigmoid_acc = train(sigmoidMLP, criterion, sgd, data_train, max_epoch, batch_size, disp_freq)


# In[16]:


test(sigmoidMLP, criterion, data_test, batch_size, disp_freq)


# ## 1.2 MLP with Euclidean Loss and ReLU Activation Function
# Build and train a MLP contraining one hidden layer with 128 units using ReLU activation function and Euclidean loss function.
# 
# ### TODO
# Before executing the following code, you should complete **layers/relu_layer.py**.

# In[8]:


from layers import ReLULayer

reluMLP = Network()
# TODO build ReLUMLP with FCLayer and ReLULayer
reluMLP.add(FCLayer(784, 128))
reluMLP.add(ReLULayer())
reluMLP.add(FCLayer(128, 10))


# In[ ]:


reluMLP, relu_loss, relu_acc = train(reluMLP, criterion, sgd, data_train, max_epoch, batch_size, disp_freq)


# In[ ]:


test(reluMLP, criterion, data_test, batch_size, disp_freq)


# ## Plot

# In[ ]:


plot_loss_and_acc({'Sigmoid': [sigmoid_loss, sigmoid_acc],
                   'relu': [relu_loss, relu_acc]})


# ## 2. MLP with Softmax Cross-Entropy Loss
# In part-2, you need to train a MLP with **Softmax Cross-Entropy Loss**.  
# **Sigmoid Activation Function** and **ReLU Activation Function** will be used respectively again.
# ### TODO
# Before executing the following code, you should complete **criterion/softmax_cross_entropy_loss.py**.

# In[9]:


from criterion import SoftmaxCrossEntropyLossLayer

criterion = SoftmaxCrossEntropyLossLayer()

sgd = SGD(learning_rate_SGD, weight_decay)


# ## 2.1 MLP with Softmax Cross-Entropy Loss and Sigmoid Activation Function
# Build and train a MLP contraining one hidden layer with 128 units using Sigmoid activation function and Softmax cross-entropy loss function.

# In[10]:


sigmoidMLP = Network()
# Build MLP with FCLayer and SigmoidLayer
# 128 is the number of hidden units, you can change by your own
sigmoidMLP.add(FCLayer(784, 128))
sigmoidMLP.add(SigmoidLayer())
sigmoidMLP.add(FCLayer(128, 10))


# ### Train

# In[11]:


sigmoidMLP, sigmoid_loss, sigmoid_acc = train(sigmoidMLP, criterion, sgd, data_train, max_epoch, batch_size, disp_freq)


# ### Test

# In[12]:


test(sigmoidMLP, criterion, data_test, batch_size, disp_freq)


# ## 2.2 MLP with Softmax Cross-Entropy Loss and ReLU Activation Function
# Build and train a MLP contraining one hidden layer with 128 units using ReLU activation function and Softmax cross-entropy loss function.

# In[9]:


reluMLP = Network()
# Build ReLUMLP with FCLayer and ReLULayer
# 128 is the number of hidden units, you can change by your own
reluMLP.add(FCLayer(784, 128))
reluMLP.add(ReLULayer())
reluMLP.add(FCLayer(128, 10))


# In[10]:


reluMLP, relu_loss, relu_acc = train(reluMLP, criterion, sgd, data_train, max_epoch, batch_size, disp_freq)


# In[11]:


test(reluMLP, criterion, data_test, batch_size, disp_freq)


# ## Plot

# In[12]:


plot_loss_and_acc({'Sigmoid': [sigmoid_loss, sigmoid_acc],
                   'relu': [relu_loss, relu_acc]})


# ### ~~You have finished homework-2, congratulations!~~  
# 
# **Next, according to the requirements 4) of experiment report:**
# ### **You need to construct a two-hidden-layer MLP, using any activation function and loss function.**
# 
# **Note: Please insert some new cells blow (using '+' bottom in the toolbar) refer to above codes. Do not modify the former code directly.**

# In[13]:


reluMLP = Network()
# Build ReLUMLP with FCLayer and ReLULayer
# 128 is the number of hidden units, you can change by your own
reluMLP.add(FCLayer(784, 256))
reluMLP.add(ReLULayer())
reluMLP.add(FCLayer(256, 64))
reluMLP.add(ReLULayer())
reluMLP.add(FCLayer(64, 10))


# In[ ]:


reluMLP, relu_loss, relu_acc = train(reluMLP, criterion, sgd, data_train, max_epoch, batch_size, disp_freq)


# In[ ]:


test(reluMLP, criterion, data_test, batch_size, disp_freq)


# In[ ]:


plot_loss_and_acc({'relu': [relu_loss, relu_acc]})


# In[18]:


import numpy as np
a = np.array([[2,4,6,1],[1,5,2,9]])
print(a)
np.argmax(a,axis=1)

