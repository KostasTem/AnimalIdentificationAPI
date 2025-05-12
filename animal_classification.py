import numpy as np
import os
import keras.optimizers as optimizers
from keras.models import Sequential
import keras.utils as image
import keras.layers as layers
from keras import applications
from keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image
import tensorflow as tf
import io

class AnimalClassificationModel:
    def __init__(self, name, data_path, training = True) -> None:
        self.name = name
        self.data_path = data_path
        self.classes = sorted(os.listdir(self.data_path))
        self.training = training
        if len(self.classes) == 1:
            return
        self.class_dict = {}
        for index, class_name in enumerate(self.classes):
            self.class_dict[index] = class_name
        base_model = applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=(224,224,3))
        base_model.trainable = False
        layer = layers.Dense(1024, activation='relu')
        if "TopLevelClassifier" not in self.name:
            layer = layers.Dense(256, activation='relu')
        self.model = Sequential([
            layers.Input(shape=(224,224,3)),
            layers.Rescaling(scale=1./255, offset=-1),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            base_model,
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Flatten(),
            layer,
            layers.Dropout(0.4),
            layers.Dense(len(self.classes), activation='softmax')
        ], name= self.name)
        optimizer = optimizers.Adam(learning_rate=0.001)
        # Compile Model
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        if not training:
            self.model.load_weights(f"{name}.h5")
        else:
            self.train_ds = image.image_dataset_from_directory(self.data_path, image_size=(224,224), label_mode="categorical", batch_size=32, shuffle=True, subset='training', validation_split=0.1, seed = 42)
            self.valid_ds = image.image_dataset_from_directory(self.data_path, image_size=(224,224), label_mode="categorical", batch_size=32, shuffle=True, subset='validation', validation_split=0.1, seed = 42)
    def train(self) -> None:
        if self.training:
            callbacks = [EarlyStopping(patience=2, restore_best_weights=True), ModelCheckpoint(self.name + ".h5", save_best_only=True)]
            self.model.fit(self.train_ds, epochs=10, validation_data=self.valid_ds, callbacks=callbacks)
    def evaluate(self, data_path) -> None:
        self.test_ds = image.image_dataset_from_directory(data_path, image_size=(224,224), label_mode="categorical", batch_size=32, shuffle=True, subset='training', validation_split=0.1, seed = 42)
        resnet50v2_loss, resnet50v2_acc = self.model.evaluate(self.test_ds)
        print(f"Loss: {resnet50v2_loss} || Accuraccy: {resnet50v2_acc}")
    def predict(self, file) -> str:
        test_image = Image.open(io.BytesIO(file))#image.load_img(path,target_size=(224,224))
        test_image = test_image.resize((224,224),resample=Image.NEAREST)
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis=0)
        prediction = self.model.predict(test_image)
        print(self.class_dict[np.argmax(prediction)])
        if prediction.max() < 0.5:
            return None
        return self.class_dict[np.argmax(prediction)]


