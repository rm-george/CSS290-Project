import tensorflow as tf
#importing this to stop some system errors from running the model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

# Directory containing images
directory_path = "\\random\\path"  # Replace with your directory path

images = []
filenames = []

try:
    # Get list of files
    file_list = os.listdir(directory_path)

    # Load each image
    for filename in file_list:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Construct full file path
            file_path = os.path.join(directory_path, filename)

            # Load image and convert to array (adjust target_size as needed)
            img = load_img(file_path, target_size=(224, 224))
            img_array = img_to_array(img)

            # Store image and filename
            images.append(img_array)
            filenames.append(filename)

    # Convert list to numpy array and preprocess
    images = np.array(images)
    images = preprocess_input(images)  # Apply appropriate preprocessing

    print(f"Loaded {len(images)} images for processing")

except FileNotFoundError:
    print("The directory was not found")```

#This is a basic dataset implemented with the tensor flow keras API which is a bunch of photos of handwriting
mnsit = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnsit.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#normalizing the data since the data is between 0 and 255
x_train, x_test = x_train / 255.0, x_test / 255.0

#Code for the model
#Will read up on theory to further explain this ripping the model from a tutorial rn
#Loss is implemented later
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

#Loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr = 0.001)
metrics = ["accuracy"]

model.compile(optimizer=optim, loss=loss, metrics=metrics)

#training
batch_size = 64
epochs = 5
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,shuffle = True, verbose=1)


#evaluating the model
model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)


#creating a new model using our old model and then pasting in to make a prediction with our already trained model and then prints
#the index of the prediction that its making for the classifications
probabilityModel = keras.Sequential([
    model,
    keras.layers.Softmax()
])

predictions = probabilityModel.predict(x_test)
pred0 = predictions[0]
print(pred0)

label0 = np.argmax(pred0)
print(label0)


#model with a softmax making a prediction
predictions = model(x_test)
predictions = tf.nn.softmax(predictions)
pred0 = predictions[7]
print(pred0)
label0 = np.argmax(pred0)
print(label0)