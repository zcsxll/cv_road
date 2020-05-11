import numpy as np
import torch
import matplotlib.pyplot as plt

class Transform:
    def __init__(self):
        pass

    def __call__(self, img_x, img_label):
        # print(img_x.shape, img_label.shape)
        img_x = img_x / 127.0 - 1

        # label = np.zeros((9, img_label.shape[0], img_label.shape[1]))
        label = np.stack([img_label] * 9)

        label[0] = np.where((label[0] == 0) | (label[0] == 249) | (label[0] == 255), 1, 0)
        label[1] = np.where((label[1] == 200) | (label[1] == 204) | (label[1] == 213) | (label[1] == 209) | (label[1] == 206) | (label[1] == 207), 1, 0)
        label[2] = np.where((label[2] == 201) | (label[2] == 203) | (label[2] == 211) | (label[2] == 208), 1, 0)
        label[3] = np.where((label[3] == 216) | (label[3] == 217) | (label[3] == 215), 1, 0)
        label[4] = np.where((label[4] == 218) | (label[4] == 219), 1, 0)
        label[5] = np.where((label[5] == 210) | (label[5] == 232), 1, 0)
        label[6] = np.where(label[6] == 214, 1, 0)
        label[7] = np.where((label[7] == 202)
                            | (label[7] == 220)
                            | (label[7] == 220)
                            | (label[7] == 221)
                            | (label[7] == 222)
                            | (label[7] == 231)
                            | (label[7] == 224)
                            | (label[7] == 225)
                            | (label[7] == 226)
                            | (label[7] == 230)
                            | (label[7] == 228)
                            | (label[7] == 229)
                            | (label[7] == 233), 1, 0)
        label[8] = np.where((label[8] == 205) | (label[8] == 212) | (label[8] == 227) | (label[8] == 223) | (label[8] == 250), 1, 0)

        # for i in range(9):
        #     plt.figure()
        #     plt.imshow(label[i])
        #     plt.savefig('./label%d.jpg' % i)
        #     plt.close()

        return torch.FloatTensor(img_x), torch.FloatTensor(label)