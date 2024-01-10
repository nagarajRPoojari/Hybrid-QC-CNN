from HybridQCCNN.config.configuration import ConfigurationManager
from HybridQCCNN.pipeline import Pipeline


configer=ConfigurationManager()
config=configer.get_model_trainer_config()
pipeline = Pipeline(config=config)



print(pipeline.inference(img_path='./download.jpg'))