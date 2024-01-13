from HybridQCCNN.components.data_transformation import DataTransformation
from HybridQCCNN.entity import DataIngetionConfig
from HybridQCCNN.entity import QuantumModelTrainerConfig
import tensorflow as tf
import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers
import dask
from numpy import pi
from pennylane import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import ResNet50 , InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input







class QCNN:
    def __init__(self,window,size=4) -> None:
        self.qubits=size**2
        self.device=qml.device('default.qubit',wires=self.qubits)
        self.size=size
        self.window=window

        @qml.qnode(device=self.device , diff_method="adjoint")
        def qcnn(params):
            for i in range(self.qubits):
                qml.RY(self.window[i//self.size][i%self.size]*(pi/2),i)
            
            StronglyEntanglingLayers(params, wires=list(range(self.qubits)))
                
            return [qml.expval(qml.PauliZ(i))for i in range(self.qubits)]
        
        self.qcnn=qcnn


class MultiQCNN:
    def __init__(self,qcnn_list:[QCNN]) -> None:
        self.qnodes=[]
        for circuit in qcnn_list:
            self.qnodes.append((qml.QNode(circuit.qcnn , circuit.device)))



class QuantumModelTrainer:
    def __init__(self,
                 config:QuantumModelTrainerConfig):
        self.config=config
        self.threads=6
        
        
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


                
                
                  
                  
                
    def qcnn_trainer(self,train_img,window_size,params,parallel=True):
        final=[]
        for i in range(0,56, window_size):
            window=train_img[i:window_size]
            circuit_list=[QCNN(window=window) for _ in range(self.threads)]
            qnodes=MultiQCNN(circuit_list).qnodes
            if parallel:
                results=tuple(dask.delayed(q)(window, params) for q in qnodes)
                final.append(results)

        return final , qnodes 
    
    
    def compute_quantum_gradients(self,qnodes, next_params):
        return qml.grad(qnodes)(next_params)
            