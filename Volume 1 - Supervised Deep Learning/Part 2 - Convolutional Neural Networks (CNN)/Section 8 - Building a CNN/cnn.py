#Part 1 - Building the CNN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

#Initializing the CNN
classifier = Sequential()

# Step 1- Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation = 'relu' ))

#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Adding a second convolution layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu' ))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Step 5- Compiling the CNN
classifier.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics = ['accuracy'])

#Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit(x=training_set, validation_data = test_set, epochs = 23 )


# Part 3 - Making new predictions
import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img("dataset/single_prediction/cat_or_dog_2.jpg", target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Dog'
else:
    prediction = 'Cat'




