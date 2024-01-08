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
