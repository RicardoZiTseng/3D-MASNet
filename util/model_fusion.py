import numpy as np
import keras

CUBE_KERNEL_KEYWORD = '.conv_nxnxn'

def _fuse_kernel(kernel, gamma, std, eps=1e-5):
    b_gamma = np.reshape(gamma, (1, 1, 1, 1, kernel.shape[-1]))
    b_gamma = np.tile(b_gamma, (kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3], 1))
    b_std = np.reshape(std, (1, 1, 1, 1, kernel.shape[-1]))
    b_std = np.tile(std, (kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3], 1))
    return kernel * b_gamma / b_std

def kernel_fusion(kernels):
    # list of parameters in the order of [nxnxn, 1xnxn, nx1xn, nxnx1]
    ksize = kernels[0].shape[0]

    kernel_nxnxn = kernels[0]
    kernel_1xnxn = kernels[1]
    kernel_nx1xn = kernels[2]
    kernel_nxnx1 = kernels[3]

    kernel_nxnxn[ksize//2:ksize//2+1, 0:ksize, 0:ksize, :, :] += kernel_1xnxn
    kernel_nxnxn[0:ksize, ksize//2:ksize//2+1, 0:ksize, :, :] += kernel_nx1xn
    kernel_nxnxn[0:ksize, 0:ksize, ksize//2:ksize//2+1, :, :] += kernel_nxnx1

    return kernel_nxnxn

def deploy(original_model, deploy_model, weights_path, fused_weights_path, eps=1e-5):
    original_model.load_weights(weights_path)
    layer_names = [layer.name for layer in original_model.layers]
    conv_nxnxn_var_names = [name for name in layer_names if CUBE_KERNEL_KEYWORD in name]
    flag = True
    for conv_nxnxn_name in conv_nxnxn_var_names:
        print(conv_nxnxn_name)
        conv_nxnxn_kernel = original_model.get_layer(conv_nxnxn_name).get_weights()[0]
        conv_1xnxn_kernel = original_model.get_layer(conv_nxnxn_name.replace(CUBE_KERNEL_KEYWORD, '.conv_1xnxn')).get_weights()[0]
        conv_nx1xn_kernel = original_model.get_layer(conv_nxnxn_name.replace(CUBE_KERNEL_KEYWORD, '.conv_nx1xn')).get_weights()[0]
        conv_nxnx1_kernel = original_model.get_layer(conv_nxnxn_name.replace(CUBE_KERNEL_KEYWORD, '.conv_nxnx1')).get_weights()[0]

        bn_nxnxn = original_model.get_layer(conv_nxnxn_name.replace(CUBE_KERNEL_KEYWORD, '.bn_nxnxn')).get_weights()
        bn_1xnxn = original_model.get_layer(conv_nxnxn_name.replace(CUBE_KERNEL_KEYWORD, '.bn_1xnxn')).get_weights()
        bn_nx1xn = original_model.get_layer(conv_nxnxn_name.replace(CUBE_KERNEL_KEYWORD, '.bn_nx1xn')).get_weights()
        bn_nxnx1 = original_model.get_layer(conv_nxnxn_name.replace(CUBE_KERNEL_KEYWORD, '.bn_nxnx1')).get_weights()

        kernels = [conv_nxnxn_kernel, conv_1xnxn_kernel, conv_nx1xn_kernel, conv_nxnx1_kernel]
        gammas = [bn_nxnxn[0], bn_1xnxn[0], bn_nx1xn[0], bn_nxnx1[0]]
        betas  = [bn_nxnxn[1], bn_1xnxn[1], bn_nx1xn[1], bn_nxnx1[1]]
        means  = [bn_nxnxn[2], bn_1xnxn[2], bn_nx1xn[2], bn_nxnx1[2]]
        vars   = [bn_nxnxn[3], bn_1xnxn[3], bn_nx1xn[3], bn_nxnx1[3]]
        if flag:
            print(kernels[0].dtype, vars[0].dtype)
            flag = False
        stds   = [np.sqrt(var + eps) for var in vars]

        fused_bias = betas[0] + betas[1] + betas[2] + betas[3] - means[0] * gammas[0] / stds[0] - means[1] * gammas[1] / stds[1] \
            - means[2] * gammas[2] / stds[2] - means[3] * gammas[3] / stds[3]

        fused_kernels = [_fuse_kernel(kernels[i], gammas[i], stds[i], eps) for i in range(4)]
        fused_weights  = kernel_fusion(fused_kernels)

        # fused_weights, fused_b = kernel_fusion(kernels, gammas, betas, means, vars)
        total_weights = [fused_weights, fused_bias]

        deploy_model.get_layer(conv_nxnxn_name.replace(CUBE_KERNEL_KEYWORD, '.fused_conv')).set_weights(total_weights)
    
    for name in layer_names:
        if '_nxnxn' not in name and '_1xnxn' not in name and '_nx1xn' not in name and '_nxnx1' not in name and '.add' not in name and 'lambda' not in name and 'concatenate' not in name:
            print(name)
            deploy_model.get_layer(name).set_weights(original_model.get_layer(name).get_weights())

    deploy_model.save_weights(fused_weights_path)

