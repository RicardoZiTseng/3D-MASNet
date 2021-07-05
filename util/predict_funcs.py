import numpy as np
import tensorflow as tf
import keras.backend as K

def calculateIdx1D(vol_length, cube_length, stride):
    one_dim_pos = np.arange(0, vol_length-cube_length+1, stride)
    if (vol_length - cube_length) % stride != 0:
        one_dim_pos = np.concatenate([one_dim_pos, [vol_length-cube_length]])
    return one_dim_pos

def calculateIdx3D(vol_size, cube_size, strides):
    x_idx, y_idx, z_idx = [calculateIdx1D(vol_size[i], cube_size[i], strides[i]) for i in range(3)]
    return x_idx, y_idx, z_idx

def idx2pos(idx, vol_size):
    """
    Given a flatten idx, return the position (x, y, z) in the 3D image space.
    Args:
        idx(int):               index into flattened 3D volume
        vol_size(list or tuple of 3 int): size of 3D volume
    """
    assert len(vol_size) == 3, 'length of vol_size must be 3, but got {}'.format(len(vol_size))

    pos_x = idx / (vol_size[1] * vol_size[2])
    idx_yz = idx % (vol_size[1] * vol_size[2])
    pos_y = idx_yz / vol_size[2]
    pos_z = idx_yz % vol_size[2]
    return np.array([pos_x, pos_y, pos_z], np.int16)  # shape of [3, ]

def pos2idx(pos, vol_size):
    """
    Given a position (x, y, z) in the 3D image space, return a flattened idx.
    Args:
        pos (list or tuple of 3 int):           Position in 3D volume
        vol_size (list or tuple of 3 int):      Size of 3D volume
    """
    assert len(pos) == 3, 'length of pos must be 3, but got {}'.format(len(pos))
    assert len(vol_size) == 3, 'length of vol_size must be 3, but got {}'.format(len(vol_size))

    return (pos[0] * vol_size[1] * vol_size[2]) + (pos[1] * vol_size[2]) + pos[2]

def calculateCubeIdx3D(cube_size, vol_size, strides):
    """
    Calculating the idx of all the patches in 3D image space.
    Args:
        cube_size (list or tuple of 3 int):
        vol_size (list or tuple of 3 int):
        strides (list or tuple of 3 int):       crop stride in 3 direction
    """
    x_idx, y_idx, z_idx = calculateIdx3D(vol_size, cube_size, strides)
    pos_idx_flat = np.zeros(x_idx.shape[0] * y_idx.shape[0] * z_idx.shape[0])
    flat_idx = 0
    
    for x in x_idx:
        for y in y_idx:
            for z in z_idx:
                pos_3d = [x, y, z]
                pos_idx_flat[flat_idx] = pos2idx(pos_3d, vol_size)
                flat_idx += 1
    
    return pos_idx_flat

def predict_segmentation(input_dict, batch_size, cube_size, strides, net, gpu_num):
    """
    Calculating the Segmentation Probability Map.
    Args:
        input_dict (dict):
            "t1w": image with shape of [*vol_size, 1]
            "t2w": image with shape of [*vol_size, 1]
        batch_size (int):
        cube_size (list or tuple of 3 int):
        strides (list or tuple of 3 int):
        net (keras.Model): the trained model
    """
    assert len(cube_size) == 3, 'length of cube_size must be 3, but got {}'.format(len(cube_size))
    assert len(strides) == 3, 'length of strides must be 3, but got {}'.format(len(strides))

    t1w = input_dict['t1w']
    t2w = input_dict['t2w']

    vol_size = t1w.shape[0:3]
    mask = np.array((t1w > 0), np.int8)
    flat_idx = calculateCubeIdx3D(cube_size, vol_size, strides)
    # flat_idx_select stores non-zero cubes' idx.
    flat_idx_select = np.zeros(flat_idx.shape)
    # remove the background cubes
    for cube_idx in range(0, flat_idx.shape[0]):
        cube_pos = idx2pos(flat_idx[cube_idx], vol_size)
        mask_cube = mask[cube_pos[0]:cube_pos[0]+cube_size[0], cube_pos[1]:cube_pos[1]+cube_size[1], cube_pos[2]:cube_pos[2]+cube_size[2]]
        if np.sum(mask_cube) != 0:
            flat_idx_select[cube_idx] = 1

    flat_idx_select = np.array(flat_idx_select, np.bool)
    flat_idx = flat_idx[flat_idx_select]
    print("Valid cube number is: ", flat_idx.shape[0])

    """
    reset the batch_size according to the number of gpus
    """
    cubes_num = flat_idx.shape[0]
    if cubes_num % batch_size < gpu_num:
        batch_size = batch_size - 1

    segmentation_predict = np.zeros(shape=(1, *vol_size, 4))
    segmentation_count = np.zeros(shape=(1, *vol_size, 1))

    batch_idx = 0
    while batch_idx < flat_idx.shape[0]:
        t1w_cubes = []
        t2w_cubes = []
        if batch_idx + batch_size < flat_idx.shape[0]:
            cur_batch_size = batch_size
        else:
            cur_batch_size = flat_idx.shape[0] - batch_idx

        for slices in range(0, cur_batch_size):
            cube_pos = idx2pos(flat_idx[batch_idx+slices], vol_size)
            t1w_cube = np.expand_dims(t1w[cube_pos[0]:cube_pos[0]+cube_size[0],
                                          cube_pos[1]:cube_pos[1]+cube_size[1],
                                          cube_pos[2]:cube_pos[2]+cube_size[2], :], axis=0)
            t2w_cube = np.expand_dims(t2w[cube_pos[0]:cube_pos[0]+cube_size[0],
                                          cube_pos[1]:cube_pos[1]+cube_size[1],
                                          cube_pos[2]:cube_pos[2]+cube_size[2], :], axis=0)
            t1w_cubes.append(t1w_cube)
            t2w_cubes.append(t2w_cube)
        
        t1w_cubes = np.concatenate(t1w_cubes, axis=0)
        t2w_cubes = np.concatenate(t2w_cubes, axis=0)

        input_cubes = np.concatenate([t1w_cubes, t2w_cubes], axis=-1)

        res_cubes = net.predict_on_batch(input_cubes)

        for slices in range(0, cur_batch_size):
            cube_pos = idx2pos(flat_idx[batch_idx+slices], vol_size)
            segmentation_predict[:,cube_pos[0]:cube_pos[0]+cube_size[0],
                                   cube_pos[1]:cube_pos[1]+cube_size[1],
                                   cube_pos[2]:cube_pos[2]+cube_size[2], :] +=res_cubes[slices]
            segmentation_count[:,  cube_pos[0]:cube_pos[0]+cube_size[0],
                                   cube_pos[1]:cube_pos[1]+cube_size[1],
                                   cube_pos[2]:cube_pos[2]+cube_size[2], :] += 1.0

        batch_idx += cur_batch_size

    # remove 0 count areas
    segmentation_count += (segmentation_count == 0)
    segmentation_predict = segmentation_predict / segmentation_count

    return segmentation_predict[0]
