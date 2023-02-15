import pandas as pd
from sensor.constant.training_pipeline import SCHEMA_FILE_PATH

import yaml

def read_yaml_file(file_path: str) -> dict:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)


anil = read_yaml_file(SCHEMA_FILE_PATH)

print(len(anil["columns"]))
