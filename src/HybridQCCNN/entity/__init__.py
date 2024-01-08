from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngetionConfig:
    root_dir:Path 
    local_data_file:Path
    unzip_dir:Path