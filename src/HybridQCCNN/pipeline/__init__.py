import tensorflow as tf
from HybridQCCNN.constants import *
from HybridQCCNN.components import *
from HybridQCCNN.entity import *
from HybridQCCNN.utils.common import read_yaml , create_directories
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# pipeline only for inferencing
class Pipeline:
    def __init__(self,config:ClassicalModelTrainerConfig) -> None:
        self.config=config
        model_path=config.model_ckpt
        self.loaded_model = tf.keras.models.load_model(model_path+'ResNet50.h5')
        self.classes=sorted(os.listdir('dataset/data_ingestion/'))
        
    def inference(self,img_path=None,image=None):
        if img_path != None:
            original_image = Image.open(img_path)
            resized_image = original_image.resize(self.config.img_size)
            image=np.array(resized_image)
        img=image[np.newaxis , :]
        res=self.loaded_model(img)
        id=np.argmax(res)
        return self.classes[id]
            
