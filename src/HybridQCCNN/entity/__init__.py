from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngetionConfig:
    root_dir:Path 
    local_data_file:Path
    unzip_dir:Path
    
    
@dataclass
class ClassicalModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: str
    model_name: str
    num_classes:int
    num_train_epochs: int
    batch_size: int
    loss: str
    optimizer: str
    metrics: str
    img_size: [int]
    
    
@dataclass
class QuantumModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: str
    model_name: str
    num_classes:int
    num_train_epochs: int
    batch_size: int
    img_size: [int]