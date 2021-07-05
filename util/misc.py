import json
import numpy as np

def cosine_annealing(epoch, lr_init, lr_min):
    return ((lr_init-lr_min)/2)*(np.cos(np.pi*(np.mod(epoch-1,50)/50))+1)+lr_min

class Params(object):
    def __init__(self, param):
        if not isinstance(param, dict):
            raise ValueError("Wrong value type, expected `dict`, but got {}".format(type(param)))
        self.param = param
    
    def __getattr__(self, name):
        return self.param[name]

def save_params(params: Params, path):
    param_dict = params.param
    param_dict = json.dumps(param_dict)
    f = open(path, 'w')
    f.write(param_dict)
    f.close()
