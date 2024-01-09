from HybridQCCNN.components.data_transformation import DataTransformation
from HybridQCCNN.entity import DataIngetionConfig
from HybridQCCNN.entity import ClassicalModelTrainerConfig
import tensorflow as tf
from tensorflow.keras.applications import ResNet50 , InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input





class ClassicalModelTrainer:
    def __init__(self,
                 config:ClassicalModelTrainerConfig):
        self.config=config
        
        
    def build_model(self):
        if self.config.model_name=='ResNet50':
            base_model=ResNet50(weights='imagenet')
        elif self.config.model_name=='InceptionV3':
            base_model=InceptionV3(weights='imagenet')
            
            
        rescale = preprocessing.Rescaling(1./255)
        
        
        inputs = Input(shape=(self.config.img_size[0],self.config.img_size[1],3))
        x = rescale(inputs)
        x=base_model(x)
        x = layers.Dense(256, activation='relu')(x)
        predictions = layers.Dense(self.config.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs=inputs, outputs=predictions)

    def train(self, train_data):
        model=self.model
        model.compile(optimizer=self.config.optimizer, loss=self.config.loss, metrics=[self.config.metrics])
        
        model.fit(train_data , epochs=self.config.num_train_epochs,steps_per_epoch=len(train_data))