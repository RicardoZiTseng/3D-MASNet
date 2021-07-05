import sys
import os
from models.dunet import DUNetMixACB
from util.model_fusion import deploy
from util.prep import maybe_mkdir_p


def main(params):
    ori_model = params.ori_model
    net = DUNetMixACB(deploy=False)
    model = net.build_model()
    net.switch_to_deploy()
    deploy_model = net.build_model()
    print("Total {} model(s) need to be fused.".format(len(ori_model)))

    for i in range(len(ori_model)):
        model_i_path = os.path.abspath(ori_model[i])
        print("Fusing model: {} ...".format(model_i_path))
        deploy_path = model_i_path.replace(".h5", "_deploy.h5")
        deploy(model, deploy_model, model_i_path, deploy_path, 1e-3)
        print("Fused model has been stored in {}".format(deploy_path))
