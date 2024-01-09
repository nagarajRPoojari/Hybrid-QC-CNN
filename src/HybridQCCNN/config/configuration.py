from HybridQCCNN.constants import *
from HybridQCCNN.components import *
from HybridQCCNN.entity import *
from HybridQCCNN.utils.common import read_yaml , create_directories

class ConfigurationManager:
    def __init__(self,
                 config_file_path=CONFIG_FILE_PATH,
                 params_file_path=PARAMS_FILE_PATH):
        self.config=read_yaml(config_file_path)
        self.params=read_yaml(params_file_path)

        
    def get_data_ingetion_config(self)-> DataIngetionConfig:
        
        config=self.config.data_ingetion
        
        create_directories([config.root_dir])
        data_ingetion_config=DataIngetionConfig(
            root_dir=config.root_dir,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingetion_config    
    
    def get_model_trainer_config(self)-> ClassicalModelTrainerConfig:
        
        config=self.config.classical_model_config
        params=self.params.ClassicalTrainingArguments
        
        create_directories([config.root_dir])
        model_trainer_config = ClassicalModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_ckpt = config.model_ckpt,
            num_train_epochs = params.num_train_epochs,
            batch_size = params.train_batch_size,
            model_name= config.model_name,
            num_classes=config.num_classes,
            optimizer= params.optimizer,
            loss=params.loss,
            metrics= params.metrics,
            img_size= config.img_size

        )
        return model_trainer_config 
    
    def get_quantum_model_trainer_config(self)-> QuantumModelTrainerConfig:
        
        config=self.config.classical_model_config
        params=self.params.ClassicalTrainingArguments
        
        create_directories([config.root_dir])
        model_trainer_config = QuantumModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_ckpt = config.model_ckpt,
            num_train_epochs = params.num_train_epochs,
            batch_size = params.train_batch_size,
            model_name= config.model_name,
            num_classes=config.num_classes,
            img_size= config.img_size

        )
        return model_trainer_config 