---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Implementing and experimenting a simple CNN on MNIST

  We will work through the material and notebooks about building a simple CNN for handwritten digit classification.
  
  In this notebook, you will experience the following:
* why CNNs are used
* how to load and prepare the MNIST dataset
* how to construct a CNN model step by step
* how to train the model
* how to use the trained model to make predictions
* how to observe the results of the intermediate steps of the model

Finally some excercises for you to solve through thinking and hands-on activity.


# Why use CNN?


The details of CNN concepts and algorithms have been introduced in Block 2 of the teaching material. Here we give you a brief recap. A fully connected feed-forward neural network, e.g. a multilayer perceptron (MLP) that you have seen and used in Block 1, will have a very high number of trainable parameters as the size of the input image increases. For example, an input image of size 28×28 pixels will have 784 weights going into the first layer. If we use 512 neurons in the first layer, then the number of total trainable parameters (weights plus bias) is 401,920 (i.e. 512×(784+1) = 401,920). If the image size is increased to 128×128 pixels, then the number of total trainable parameters is 8,389,120 (i.e. 512×(128×128+1) = 8,389,120). An increase in the number of parameters will increase the computational complexity and thus make the network slow. The risk of overfitting also increases with an increase in the number of parameters.

Moreover, MLP neglects the spatial structure information that is key to characterising the high-level semantics of images. In MLP, the layers are typically fully connected. Each input neuron feeds into every neuron in the next layer, and each neuron in subsequent layers feeds into every neuron in the next layer. Because each neuron receives input from all the pixels in the image, the pixels that are close together are not distinguished from those that are far apart. Furthermore, the positions and directions of the same visual objects can change in different images due to translations and rotations of the objects. Thus the issue of translational variance needs to be captured. Traditional artificial neural networks such as MLP cannot tackle these issues effectively. CNNs aim to overcome these shortcomings.

A CNN uses a relatively low number of parameters compared to the fully connected feed-forward neural network for the same input image. The techniques used in CNN also increase accuracy for image-related tasks. The techniques used are convolution and pooling. The techniques are implemented in the form of neural network layers.

<!-- #region -->
## Convolution layer 


The idea of the convolution layer is to create a filter (also called kernel). The filter is used to scan across the image and create a representation of the image corresponding to the filter. In this way, we can think of the filter as a specific feature extraction mechanism for the image. In a single convolution layer, we can define more than one filter, where each of them would detect a specific feature of an image, as illustrated below (i.e. Figure 7 in the material):

![title](./pic/Example-filters.png)
 
If we use 32 different filters in the convolution layer, then we create 32 different representations, called **feature maps**, of the input image. These different feature maps in combination can help us identify the input image correctly.

How a filter operates is illustrated in Figure 11 (Section 3.1) in the teaching material. Simply speaking, we overlay a filter, which is a small matrix of weights, on top of the input matrix, e.g. starting from the top-left corner and then sliding from left to right and from top to bottom. 

![title](./pic/example-convolution.png)

The **filter** shown in the example is of size *3×3*. It is applied on a *5×5* input matrix (you can consider it as the matrix of pixels for an image). When the filter moves through the image it moves with a **stride** of **1**, i.e. the kernel covers *3×3* part of the image, does its computation, and then moves 1 step to cover another *3×3* part of the input. The computation is matrix multiplication with the value of filter and the area in the image that the kernel is currently covering. The right-hand side matrix shows the output of each step. 

There is another key concept known as **zero-padding**. It adds zeros to the borders of the input image to avoid filters moving outside the image boundary. 

*The values in the kernel are learnt during the training step.*
<!-- #endregion -->

## Pooling Layer

Pooling is used to reduce the size of the input image by summarising specific regions. Similar to the convolution kernel, the pooling layer uses a **grid**  and a **stride** which determines the movement of the grid.
The pooling operation is specified and thus does *not require learning*. Two common functions used in the pooling operation are:

* Average Pooling, which calculates the average value for each grid on the input image.

* Maximum Pooling (or Max Pooling), which calculates the maximum value for each grid on the input image.

Pooling layers operate on the feature maps. Since the convolution layer produces many feature maps of the input image, it is not necessary to maintain the high dimension of each feature map. Pooling reduces the size of the result by combining certain neighbouring values together. It also helps with translational variance by giving the same end result even if the input image is moved slightly. 



# Implementing a Simple CNN

Next, we will see how a CNN is implemented using the Keras API (part of the TensorFlow library). Firstly, we load the toolkits we need to use:

```python
# Load some toolkits we will need later
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, metrics, Sequential
from tensorflow.keras.layers import *

import tensorflow_datasets as tfds
```

```python
ROWS = 32
COLS = 32
CHANNELS = 3
INPUT_SHAPE = (ROWS, COLS, CHANNELS)

BATCH_SIZE = 64
```

### Loading and preparing the MNIST dataset from TFDS

Now we can load the data. Loading and preparing the dataset here is identical to the Block 1 notebooks. 

```python
# Loading the data
(train_data, test_data), dataset_info = tfds.load('cifar10',
    data_dir='/datasets',
    split=['train', 'test'],
    with_info=True)
train_data, test_data, dataset_info
```

```python
# Preparing the data

# define the list of class labels
class_names =  ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

class_num = len(class_names)

# Convert a tdfs mnist element into the form we require for training.
# It turns an integer representing a class integer into an one-hot vector. 
# For example, label 2 would become [0, 0, 1, 0, 0, 0, 0, 0, 0, 0].

def ds_elem_transform(elem):
    return (tf.cast(elem['image'], tf.float32) / 255, # convert pixel values to range 0-1
           tf.one_hot(elem['label'], 10) # one-hot encoding for labels, 10 choices
           )

# Transform every element of the dataset, rename the result.
train_validation_data = train_data.map(
    ds_elem_transform, num_parallel_calls=tf.data.AUTOTUNE)

# Preparing the training and validation data
# Take the elements at index 9, 19, 29... into the validation dataset
validation_data = train_validation_data.shard(10, 9)
validation_data = validation_data.batch(64)
validation_data = validation_data.cache()
validation_data = validation_data.prefetch(tf.data.AUTOTUNE)

# Create a new train_data dataset. Append the remaining shards of the train_validation_data dataset.
train_data = train_validation_data.shard(10, 8)
for i in range(8):
    train_data = train_data.concatenate(train_validation_data.shard(10, i))

train_data = train_data.cache()
train_data = train_data.shuffle(dataset_info.splits['train'].num_examples)
train_data = train_data.batch(64)
train_data = train_data.prefetch(tf.data.AUTOTUNE)

# Perform the same steps on the test dataset:
# transform each element, batch, cache, and prefetch.
test_data = test_data.map(
    ds_elem_transform, num_parallel_calls=tf.data.AUTOTUNE)
test_data = test_data.batch(64)
test_data = test_data.cache()
test_data = test_data.prefetch(tf.data.AUTOTUNE)

```

Next, let us take a look at what the data we have loaded look like.


```python
train_data, validation_data, test_data
```

How much data do we have?

```python
len(train_data), len(validation_data), len(test_data)
```

That's 704 _batches_ of training data, with 64 images per batch, meaning there are just over 45,000 images to train from.


### Viewing some pictures
 
We use the `matploblib` library to show 25 sample images in the `train_data` we have loaded. Again, the method here is identical to the Block 1 notebooks.

1. We first use `plt.figure` make a canvas with size 10×10 
2. We draw 25 pictures in the canvas. The **subplot** API is used to control where we draw on the canvas

```python
# Get the first batch. 
# (This may take a moment as it reads and caches the Dataset)
sample_imgs, sample_labels = train_data.as_numpy_iterator().next()

# The canvas size is first initialized by setting the parameter figsize in function plt.figure()
plt.figure(figsize=(10,10)) 
for i in range(25):
    plt.subplot(5,5,i+1)  # draw five rows and five columns in canvas
    plt.imshow(sample_imgs[i], cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title(class_names[np.argmax(sample_labels[i])])
plt.show()
```

### Building a CNN step by step

Our image-classification model is a linear stack of layers,

The `Sequential` constructor takes a list of `Layers`. As shown in the figure below, we'll use four types of layers for our CNN: convolution, max pooling, dropout, and softmax. In the example structure, one convolution layer is used. The convolution layer uses 32 *3×3* filters with stride 1 and padding, resulting in 32 *32×32* feature maps. Then, applying *2×2* max-pooling, the size of the feature maps are reduced to *16×16*. The dropout layer will randomly pass of those features to the next layer, choosing different features each time (this reduces overfitting). The features are then flattened to a vector of size 512. The vector is fed into a fully connected layer followed by a softmax layer to make predictions. 


![title](pic/cnn-diagram.png)

```python
model = Sequential([
    Input(INPUT_SHAPE),
    Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Flatten(),
    Dense(512),
    Dense(class_num, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )


# We usually check our model by printing the model parameters through model.summary(), after defining the model.
# The model summary will show the type of model, the output of each layer and the number of parameters used by each layer.
# For example, from the input 28*28*1, through the first conv2d layer, a 26*26*8 output is obtained. 
# The number of parameters = (convolution filter size * number of channels + 1) * number of output channels.
# So, the number of parameters for this layer is (3 * 3 * 1 + 1) * 8 = 80

model.summary()
```

### Training the CNN: Loss Function, Optimizer, Metrics

Before training process starts, we configure the training process, including 3 key factors during the compilation step:
1. The optimizer. Keras provides various optimizers. A good default is the Adam gradient-based optimizer. 
2. The loss function. Since we're using a Softmax output layer, the cross-Entropy loss function is used. Keras distinguishes between `binary_crossentropy` (2 classes) and `categorical_crossentropy` (>2 classes), so we'll use the latter. 
3. Metrics. The accuracy metric is used.

Note that this is the same process as in Block 1.

```python
model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

Training a model in Keras is done by calling `fit()` and specifying hyper-parameters. 

Main parameters include:

1. The number of epochs (iterations of training over the entire training dataset).
2. The validation data, which is used during training to periodically measure the network's performance against the data that hasn't been seen before.

Running the following code on the full MNIST dataset gives us results.

```python
history = model.fit(train_data,
    validation_data=validation_data,
    epochs=10,
         )
```

Immediately, we can see the changes in accuracy over different epochs. We get to an validation accuracy of around 0.97 after 10 epochs. 

Recall that the Perceptron-based model from Block 1 achieved an accuracy of around 90%. The CNN model seems to work much better.

We can plot the changes in training and validation accuracy and loss as below.

```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()  # Automatic detection of elements to be shown in the legend
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.figure()

plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()  # Automatic detection of elements to be shown in the legend
plt.figure()
```

This impression is reflected in the overall accuracy and loss results. The model performs better on the validation data than the training data. Let's see what happens if we continue training for anther 10 epochs.

```python
history = model.fit(train_data,
    validation_data=validation_data,
    initial_epoch=10,
    epochs=30 )
```

```python
acc += history.history['accuracy']
val_acc += history.history['val_accuracy']
loss += history.history['loss']
val_loss += history.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure()
```

From the plots, we can see that this simple CNN structure starts to converge after 10 epochs and reaches a validation accuracy around 0.97.


### Making predictions on test data

We can now use the model to make predictions on the test data. We will:

1. use the trained model to make predictions on the test set
2. view the prediction results, and find the probabilities of the class labels to which each image belongs
3. select the label with the highest probability
4. display the correct labels and predicted labels of sample images in the test set.

Again, the process is the same as in the Block 1 notebooks.

```python
test_predictions = model.predict(test_data)
test_predictions.shape
```

```python
test_predictions[1]
```

```python
predict_labels = np.argmax(test_predictions, axis=1)
```

```python
# View the true and predicted labels of sample images
plt.figure(figsize=(15,10))
test_imgs, test_labels = test_data.as_numpy_iterator().next()

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_imgs[i], cmap=plt.cm.binary)
    p_class = predict_labels[i]
    a_class = np.argmax(test_labels[i])
    plt.title(f"P: {class_names[p_class]} (A: {class_names[a_class]})",
                                  color=("green" if p_class == a_class else "red"))
plt.show()

```

Finally, we can evaluate the overall accuracy of the model on test data.

```python
model_results = model.evaluate(test_data, return_dict=True)
model_results
```

So far, this simple CNN model has worked well. Now let's drill down into the model and take a closer look at what's going on at each level of our model.


### Exploring intermediate layers of the model

Note that this is new content, and a new way of looking at models and what they're doing inside. We will:

1. create an "intermediate" model, with the original input of the trained model as input and the output of the first convolutional layer of the model as output.
2. make predictions with the intermediate model.
3. view the original input.
4. look at the 8 channels generated after the input image passes through the first convolutional layer.

Now, a reminder of the model summary and different layers in the model.

```python
model.summary()
```

```python
model.layers
```

Now let us create an intermediate model consisting of the original input and the first layer of the original model. 

We check the summary of the intermediate model. You can also check the layer attributes of the intermediate model and compare them with the first layer of the original model (note the address when looking at the layers attribute: you should see the same IDs after the word `at`).

```python
intermediate_layer_model = keras.Model(inputs=model.inputs,
                                       outputs=model.layers[0].output)
intermediate_layer_model.summary()

intermediate_layer_model.layers
```

We fetch a test batch of 64 images, and show the first one.

```python
test_batch, test_batch_labels = test_data.as_numpy_iterator().next()
test_batch.shape
```

```python
# ploting the original image
plt.figure()
# test_batch has a shape 64*28*28*1, here we use test_batch[0] to pick the first image, 
plt.imshow(test_batch[0]) 
# plt.savefig('./exp_pic/activity_2_figure_3.svg')
```

Now we can show some sample outputs of the intermediate model.

```python
intermediate_output = intermediate_layer_model(test_batch)
```

```python
# Plotting the 8 intermediate feature maps generated by the 8 filters in the first convolution layer

plt.figure(figsize=(10, 5))

for i in range(32):
    plt.subplot(4, 8, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(intermediate_output[0,:,:,i]) 
    plt.xlabel(f'Filter {i}', fontsize=10)
```

You should see that different feature maps pick out different elements of the original image, such as horizontal, vertical, and angled lines. But exactly what features are detected will depend on how your network initialised and trained itself.

<!-- #region tags=["style-activity"] -->
# Exercise
Examine the filter outputs for a different image in the batch.
<!-- #endregion -->

```python tags=["style-solution"]
# Your solution here
# Add additional cells as needed
```

<!-- #region tags=["style-student"] -->
### Example solution
<!-- #endregion -->

```python tags=["style-student"]
figure_num = 2

plt.figure()
plt.imshow(test_batch[figure_num]) 
plt.show()

plt.figure(figsize=(10, 5))

for i in range(32):
    plt.subplot(4, 8, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(intermediate_output[figure_num,:,:,i]) 
    plt.xlabel(f'Filter {i}', fontsize=10)
```

<!-- #region tags=["style-student"] -->
### End solution
<!-- #endregion -->

Following the previous steps, let's see what happens after the first `MaxPooling` layer.

```python
intermediate_layer_model = keras.Model(inputs=model.inputs,
                                       outputs=model.layers[1].output)
intermediate_output = intermediate_layer_model(test_batch)

plt.figure(figsize=(10, 5))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(intermediate_output[0, :, :, i])
    plt.xlabel(f'Filter {i}', fontsize=10)
```

You should see that the max pooling layer gives a "lower-resolution image" of the corresponding filter. 

Let's continue to explore the output of the second convolutional layer and the output of the second max pooling layer.

```python
intermediate_layer_model = keras.Model(inputs=model.inputs,
                                       outputs=model.layers[3].output)
intermediate_output = intermediate_layer_model(test_batch)

plt.figure(figsize=(10, 5))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(intermediate_output[0,:,:,i])
    plt.xlabel(f'Filter {i}', fontsize=10)
```

```python
intermediate_layer_model = keras.Model(inputs=model.inputs,
                                       outputs=model.layers[4].output)
intermediate_output = intermediate_layer_model(test_batch)

plt.figure(figsize=(10, 5))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(intermediate_output[0,:,:,i])
    plt.xlabel(f'Filter {i}', fontsize=10)
```

<!-- #region tags=["style-activity"] -->
# Exercise
<!-- #endregion -->

```python
model3 = Sequential([
    Input((32, 32, 3)),
    Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(class_num, activation='softmax')
])

model3.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )
```

```python
model3.summary()
```

```python
history = model3.fit(train_data,
    validation_data=validation_data,
    epochs=30,
    verbose=0
         )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figure()
```

```python
model3_results = model3.evaluate(test_data, return_dict=True)
model3_results
```

```python

```

```python

```

```python
list(enumerate(model3.layers))
```

Now let us create an intermediate model consisting of the original input and the first layer of the original model. 

We check the summary of the intermediate model. You can also check the layer attributes of the intermediate model and compare them with the first layer of the original model (note the address when looking at the layers attribute: you should see the same IDs after the word `at`).

```python
intermediate_layer_model = keras.Model(inputs=model3.inputs,
                                       outputs=model3.layers[0].output)
intermediate_layer_model.summary()
```

We fetch a test batch of 64 images, and show the first one.

```python
test_batch, test_batch_labels = test_data.as_numpy_iterator().next()
test_batch.shape
```

```python
# ploting the original image
plt.figure()
# test_batch has a shape 64*28*28*1, here we use test_batch[0] to pick the first image, 
plt.imshow(test_batch[0]) 
# plt.savefig('./exp_pic/activity_2_figure_3.svg')
```

Now we can show some sample outputs of the intermediate model.

```python
intermediate_output = intermediate_layer_model(test_batch)
```

```python
# Plotting the 8 intermediate feature maps generated by the 8 filters in the first convolution layer

plt.figure(figsize=(10, 5))

for i in range(32):
    plt.subplot(4, 8, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(intermediate_output[0,:,:,i]) 
    plt.xlabel(f'Filter {i}', fontsize=10)
```

```python

```

```python

```

```python
intermediate_layer_model2 = keras.Model(inputs=model3.inputs,
                                       outputs=model3.layers[2].output)
intermediate_layer_model2.summary()
```

```python
intermediate_output2 = intermediate_layer_model2(test_batch)
```

```python
# Plotting the 8 intermediate feature maps generated by the 8 filters in the first convolution layer

plt.figure(figsize=(10, 5))

for i in range(32):
    plt.subplot(4, 8, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(intermediate_output2[0,:,:,i]) 
    plt.xlabel(f'Filter {i}', fontsize=10)
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

# Conclusion


You can further explore CNN by creating one or two your own models. For example, you may see how the performance changes if you remove the max-pooling layers and dropout layers; or if you change the number of filters, e.g., to 4. 

```python

```
