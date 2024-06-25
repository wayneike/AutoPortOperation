

### https://colab.research.google.com/drive/18-cZLLZq8W5u9mi5w9-9tybU_LvFvh1i?usp=sharing

Training dataset can be downloaded from here:
Dataset from https://www.kaggle.com/datasets/clorichel/boat-types-recognition



# Automating Port Operations

## Project Statement:
Marina Pier Inc. is leveraging technology to automate their operations on the San Francisco port.

The company’s management has set out to build a bias-free/ corruption-free automatic system that reports & avoids faulty situations caused by human error.

Examples of human error include misclassifying the correct type of boat. The type of boat that enters the port region is as follows.
* Buoy
* Cruise_ship
* Ferry_boat
* Freight_boar
* Gondola
* Inflatable_boat
* Kayak
* Paper_boat
* Sailboat

Marina Pier wants to use Deep Learning techniques to build an automatic reporting system that recognizes the boat. The company is also looking to use a transfer learning approach of any lightweight pre-trained model in order to deploy in mobile devices.

As a deep learning engineer, your task is to:
1. Build a CNN network to classify the boat.
1. Build a lightweight model with the aim of deploying the solution on a mobile device using transfer learning. You can use any lightweight pre-trained model as the initial (first) layer. MobileNetV2 is a popular lightweight pre-trained model built using Keras API.

## Dataset and Data Description:
boat_type_classification_dataset.zip

The dataset contains images of 9 types of boats. It contains a total of 1162 images. The training images are provided in the directory of the specific class itself.

Classes:
* ferry_boat
* gondola
* sailboat
* cruise_ship
* kayak
* inflatable_boat
* paper_boat
* buoy
* freight_boat

## Perform the following steps:

1. Build a CNN network to classify the boat.__
    * Split the dataset into train and test in the ratio 80:20, with shuffle and random state=43.
    * Use Keras ImageDataGenerator to initialize the train generator with validation_split=0.2 and test generator. Generators are required to avoid out of memory issues while training. -- NOTE: ImageDataGenerator has been deprecataed in favor of tf.keras.utils.image_dataset_from_directory.
   https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
    * Both generators will be initialized with data normalization.
(Hint: rescale=1./255).
    * Load train, validation and test dataset in batches of 32 using the generators initialized in the above step.
    * Build a CNN network using Keras with the following layers
        - Cov2D with 32 filters, kernel size 3,3, and activation relu, followed by MaxPool2D
        - Cov2D with 32 filters, kernel size 3,3, and activation relu, followed by MaxPool2D
        - GLobalAveragePooling2D layer
        - Dense layer with 128 neurons and activation relu
        - Dense layer with 128 neurons and activation relu
        - Dense layer with 9 neurons and activation softmax.
    * Compile the model with Adam optimizer, categorical_crossentropy loss, and with metrics accuracy, precision, and recall.
    * Train the model for 20 epochs and plot training loss and accuracy against epochs.
    * Evaluate the model on test images and print the test loss and accuracy.
    * Plot heatmap of the confusion matrix and print classification report.
<br><br>
1. Build a lightweight model with the aim of deploying the solution on a mobile device using transfer learning. You can use any lightweight pre-trained model as the initial (first) layer. MobileNetV2 is a popular lightweight pre-trained model built using Keras API.
    * Split the dataset into train and test datasets in the ration 70:30, with shuffle and random state=1.
    * Use Keras ImageDataGenerator to initialize the train generator with validation_split=0.2 and test generator. Generators are required to avoid out-of-memory issues while training. -- NOTE: ImageDataGenerator has been deprecataed in favor of tf.keras.utils.image_dataset_from_directory.
   https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
    * Both generators will be initialized with data normalization.
(Hint: rescale=1./255).
    * Load train, validation and test datasets in batches of 32 using the generators initialized in the above step.
    * Build a CNN network using Keras with the following layers.
        - Load MobileNetV2 - Light Model as the first layer
(Hint: Keras API Doc)
        - GLobalAveragePooling2D layer
        - Dropout(0.2)
        - Dense layer with 256 neurons and activation relu
        - BatchNormalization layer
        - Dropout(0.1)
        - Dense layer with 128 neurons and activation relu
        - BatchNormalization layer
        - Dropout(0.1)
        - Dense layer with 9 neurons and activation softmax
    * Compile the model with Adam optimizer, categorical_crossentropy loss, and metrics accuracy, Precision, and Recall.
    * Train the model for 50 epochs and Early stopping while monitoring validation loss.
    * Evaluate the model on test images and print the test loss and accuracy.
    * Plot Train loss Vs Validation loss and Train accuracy Vs Validation accuracy.
<br><br>
1. Compare the results of both models built in steps 1 and 2 and state your observations.