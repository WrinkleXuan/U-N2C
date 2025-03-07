from .build import build_loader_train as _build_loader_train
from .build import build_loader_test as _build_loader_test

def build_loader(config):
    if config.MODE=='train':
        return _build_loader_train(config)
    else:
        return _build_loader_test(config)
 
