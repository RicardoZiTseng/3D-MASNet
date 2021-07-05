import math
import os
import numpy as np
import nibabel as nib
from random import random

from util.prep import image_norm, make_onehot_label

class DataGen(object):
    def __init__(self, file_path, id_list):
        self.batch_size = 16
        self.cube_size = 32
        self.file_list = []
        for ids in id_list:
            datas = {}
            subject_name = 'subject-{}-'.format(ids)
            print("load image file: {}".format(subject_name))
            T1 = os.path.join(file_path, subject_name + 'T1.hdr')
            T2 = os.path.join(file_path, subject_name + 'T2.hdr')
            label = os.path.join(file_path, subject_name + 'label.hdr')
            t1_data = nib.load(T1).get_data()
            t2_data = nib.load(T2).get_data()
            mask = np.array(t1_data > 0, dtype=np.int8)
            label_data = make_onehot_label(nib.load(label).get_data(), 4)
            t1_data = image_norm(t1_data)
            t2_data = image_norm(t2_data)
            datas['images'] = np.concatenate([t1_data, t2_data], axis=-1)
            datas['label'] = label_data
            datas['mask'] = mask
            self.file_list.append(datas)

    def make_gen(self):
        while True:
            curr_batch_idx = 0
            images_cubes = []
            label_cubes = []
            while curr_batch_idx < self.batch_size:
                file_idx = np.random.randint(0, len(self.file_list))
                random_file = self.file_list[file_idx]
                h, w, d, _ = random_file['images'].shape

                while True:
                    random_hidx = np.random.randint(0, h-self.cube_size)
                    random_widx = np.random.randint(0, w-self.cube_size)
                    random_didx = np.random.randint(0, d-self.cube_size)
                    mask_cube = random_file['mask'][random_hidx:random_hidx+self.cube_size,
                                                    random_widx:random_widx+self.cube_size,
                                                    random_didx:random_didx+self.cube_size, :]
                    if np.sum(mask_cube) != 0:
                        break

                random_images_cube = np.expand_dims(random_file['images'][random_hidx:random_hidx+self.cube_size,
                                                                        random_widx:random_widx+self.cube_size,
                                                                        random_didx:random_didx+self.cube_size, :], axis=0)
                random_label_cube = np.expand_dims(random_file['label'][random_hidx:random_hidx+self.cube_size,
                                                                        random_widx:random_widx+self.cube_size,
                                                                        random_didx:random_didx+self.cube_size, :], axis=0)
                    
                images_cubes.append(random_images_cube)
                label_cubes.append(random_label_cube)
                curr_batch_idx += 1

            images_cubes = np.concatenate(images_cubes, axis=0)
            label_cubes = np.concatenate(label_cubes, axis=0)

            yield (
                {'input': images_cubes},
                {'output': label_cubes}
            )
