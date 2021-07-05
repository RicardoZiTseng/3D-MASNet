import sys
import json

class Params(object):
    def __init__(self, param):
        if not isinstance(param, dict):
            raise ValueError("Wrong value type, expected `dict`, but got {}".format(type(param)))
        self.param = param
    
    def __getattr__(self, name):
        return self.param[name]

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        params = Params(json.load(f))
        if params.task == 'train':
            from .train import main as train_main
            train_main(params)
        elif params.task == 'predict':
            from .predict import main as predict_main
            predict_main(params)
        elif params.task == 'deploy':
            from .deploy import main as deploy_main
            deploy_main(params)
        else:
            raise ValueError("Wrong params `task`, expected `train` or `predict`, but got `{}`".format(params.task))