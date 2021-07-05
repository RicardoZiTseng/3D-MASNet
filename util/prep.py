import numpy as np
import os

def maybe_mkdir_p(directory):
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i+1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i+1]))
            except FileExistsError:
                # this can sometimes happen when two jobs try to create the same directory at the same time,
                # especially on network drives.
                print("WARNING: Folder %s already existed and does not need to be created" % directory)

def make_onehot_label(label, num_classes):
    onehot_encoding = []
    for c in range(num_classes):
        onehot_encoding.append(label == c)
    onehot_encoding = np.concatenate(onehot_encoding, axis=-1)
    onehot_encoding = np.array(onehot_encoding, dtype=np.int16)
    return onehot_encoding

def _calIdx1D(vol_length, cube_length, stride):
    one_dim_pos = np.arange(0, vol_length-cube_length+1, stride)
    if (vol_length - cube_length) % stride != 0:
        one_dim_pos = np.concatenate([one_dim_pos, [vol_length-cube_length]])
    return one_dim_pos

def _calIdx3D(vol_size, cube_size, strides):
    x_idx, y_idx, z_idx = [_calIdx1D(vol_size[i], cube_size[i], strides[i]) for i in range(3)]
    return x_idx, y_idx, z_idx

def crop(data, x, y, z, cube_size):
    assert len(cube_size) == 3
    return data[:, x:x+cube_size[0],
                   y:y+cube_size[1],
                   z:z+cube_size[2], :]

def image_norm(img):
    mask = (img > 0)
    mean = np.mean(img[mask])
    std = np.std(img[mask])
    return ((img - mean) / std) * mask
    # return (img - mean) / std

def cut_edge(data, keep_margin):
    '''
    function that cuts zero edge
    # Ricardo: calculating the margin number of non-zero area.
    '''
    D, H, W, _ = data.shape
    # print(D, H, W)
    D_s, D_e = 0, D - 1
    H_s, H_e = 0, H - 1
    W_s, W_e = 0, W - 1

    while D_s < D:
        if data[D_s].sum() != 0:
            D_s -= 1
            break
        D_s += 1
    while D_e > D_s:
        if data[D_e].sum() != 0:
            D_e += 1
            break
        D_e -= 1
    while H_s < H:
        if data[:, H_s].sum() != 0:
            H_s -= 1
            break
        H_s += 1
    while H_e > H_s:
        if data[:, H_e].sum() != 0:
            H_e += 1
            break
        H_e -= 1
    while W_s < W:
        if data[:, :, W_s].sum() != 0:
            W_s -= 1
            break
        W_s += 1
    while W_e > W_s:
        if data[:, :, W_e].sum() != 0:
            W_e += 1
            break
        W_e -= 1

    if keep_margin != 0:
        D_s = max(0, D_s - keep_margin)
        D_e = min(D - 1, D_e + keep_margin)
        H_s = max(0, H_s - keep_margin)
        H_e = min(H - 1, H_e + keep_margin)
        W_s = max(0, W_s - keep_margin)
        W_e = min(W - 1, W_e + keep_margin)

    return int(D_s), int(D_e), int(H_s), int(H_e), int(W_s), int(W_e)

def vote(segmentation_sets):
    onehot_segmentation_sets = [make_onehot_label(seg, 4) for seg in segmentation_sets]
    vote_seg = np.argmax(sum(onehot_segmentation_sets), axis=-1)
    vote_seg = np.array(np.expand_dims(vote_seg, axis=-1), dtype=np.uint8)
    return vote_seg


