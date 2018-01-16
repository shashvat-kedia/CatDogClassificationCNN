from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras import backend as K
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

K.set_image_dim_ordering('th')
classifier = Sequential()
classifier.add(Convolution2D(64,(3,3),input_shape=(3,128,128),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(64,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(activation='relu',units=128))
classifier.add(Dense(activation='sigmoid',units=1))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(
        rescale=1./255)
train = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128,128),
        batch_size=32,
        class_mode='binary')
test = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128,128),
        batch_size=20,
        class_mode='binary')
classifier.fit_generator(
        train,
        samples_per_epoch=8000,
        nb_epoch=25,
        validation_data=test,
        nb_val_samples=2000)
