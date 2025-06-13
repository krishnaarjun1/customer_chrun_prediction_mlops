import yaml

def load_config(path='src\config\config.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)
