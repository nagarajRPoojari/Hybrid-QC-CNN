from HybridQCCNN.entity import DataIngetionConfig
from HybridQCCNN.logging import logger
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataTransformation:
    def __init__(self,
                 config:DataIngetionConfig):
        
        self.config=config
    
    def data_loader(self , grayscale=False):
        image_dir=self.config.unzip_dir
        datagen = ImageDataGenerator(
            rescale=1./255,
            
        )    
        batch_size = 32
        train_data = tf.keras.preprocessing.image_dataset_from_directory(
            image_dir,
            image_size=(224, 224),
            batch_size=batch_size,
            label_mode='categorical', 
            validation_split=0.2,
            seed=42,
            subset='training',
            color_mode='grayscale' if grayscale else 'rgb'
        )
        
        self.train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return self.train_data
        