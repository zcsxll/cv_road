import numpy as np
import torch

class Transform:
    def __init__(self):
        pass

    def __call__(self, img_x, img_label):
        # print(img_x.shape, img_label.shape)
        img_x = img_x / 127.0 - 1
        img_label = np.where(img_label == 200, 1, img_label)
        img_label = np.where(img_label == 204, 1, img_label)
        img_label = np.where(img_label == 213, 1, img_label)
        img_label = np.where(img_label == 209, 1, img_label)
        img_label = np.where(img_label == 206, 1, img_label)
        img_label = np.where(img_label == 207, 1, img_label)

        img_label = np.where(img_label == 201, 2, img_label)
        img_label = np.where(img_label == 203, 2, img_label)
        img_label = np.where(img_label == 211, 2, img_label)
        img_label = np.where(img_label == 208, 2, img_label)

        img_label = np.where(img_label == 216, 3, img_label)
        img_label = np.where(img_label == 217, 3, img_label)
        img_label = np.where(img_label == 215, 3, img_label)

        img_label = np.where(img_label == 218, 4, img_label)
        img_label = np.where(img_label == 219, 4, img_label)

        img_label = np.where(img_label == 210, 5, img_label)
        img_label = np.where(img_label == 232, 5, img_label)

        img_label = np.where(img_label == 214, 6, img_label)

        img_label = np.where(img_label == 202, 7, img_label)
        img_label = np.where(img_label == 220, 7, img_label)
        img_label = np.where(img_label == 221, 7, img_label)
        img_label = np.where(img_label == 222, 7, img_label)
        img_label = np.where(img_label == 231, 7, img_label)
        img_label = np.where(img_label == 224, 7, img_label)
        img_label = np.where(img_label == 225, 7, img_label)
        img_label = np.where(img_label == 226, 7, img_label)
        img_label = np.where(img_label == 230, 7, img_label)
        img_label = np.where(img_label == 228, 7, img_label)
        img_label = np.where(img_label == 229, 7, img_label)
        img_label = np.where(img_label == 233, 7, img_label)

        img_label = np.where(img_label == 205, 8, img_label)
        img_label = np.where(img_label == 212, 8, img_label)
        img_label = np.where(img_label == 227, 8, img_label)
        img_label = np.where(img_label == 223, 8, img_label)
        img_label = np.where(img_label == 250, 8, img_label)

        img_label = np.where(img_label == 249, 0, img_label)
        img_label = np.where(img_label == 255, 0, img_label)


        ret = img_label[img_label > 8]
        assert len(ret) == 0
        # print(len(ret))

        return torch.FloatTensor(img_x), torch.FloatTensor(img_label)