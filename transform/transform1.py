import numpy as np
import torch
# import torchvision
import matplotlib.pyplot as plt
import cv2
from imgaug import augmenters as iaa
import random

# pixel augmentation
class ImageAug(object):
    def __call__(self, image, label):
        # 在0-1之间随机
        if np.random.uniform(0, 1) > 0.5:
            seq = iaa.Sequential([iaa.OneOf([
                iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)), #加一个高斯噪音
                iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3)), #锐化
                iaa.GaussianBlur(sigma=(0, 1.0))])]) #高斯模糊进行随机选择

            image = seq.augment_image(image)
        return image, label

class DeformAug(object):
    def __call__(self, image, label):
        seq = iaa.Sequential([iaa.CropAndPad(percent=(-0.05, 0.1))])
        seg_to = seq.to_deterministic()
        image = seg_to.augment_image(image)
        label = seg_to.augment_image(label)
        return image, label

class ScaleAug(object):
    def __call__(self, image, label):
        scale = random.uniform(0.7, 1.5)
        h, w, _ = image.shape
        # aug_image = image.copy()
        # aug_label = mask.copy()

        image = cv2.resize(image, (int (scale * w), int (scale * h)))
        label = cv2.resize(label, (int (scale * w), int (scale * h)))

        if scale < 1.0:
            new_h, new_w, _ = image.shape
            pre_h_pad = int((h - new_h) / 2)
            pre_w_pad = int((w - new_w) / 2)
            pad_list = [[pre_h_pad, h - new_h - pre_h_pad], [pre_w_pad, w - new_w - pre_w_pad], [0, 0]]
            image = np.pad(image, pad_list, mode="constant")
            label = np.pad(label, pad_list[:2], mode="constant")
        if scale > 1.0:
            new_h, new_w, _ = image.shape
            pre_h_crop = int ((new_h - h) / 2)
            pre_w_crop = int ((new_w - w) / 2)
            post_h_crop = h + pre_h_crop
            post_w_crop = w + pre_w_crop
            image = image[pre_h_crop:post_h_crop, pre_w_crop:post_w_crop]
            label = label[pre_h_crop:post_h_crop, pre_w_crop:post_w_crop]
        return image, label

class CutOut(object):
    def __init__(self, mask_size, p):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, image, label):
        mask_size_half = self.mask_size // 2
        offset = 1 if self.mask_size % 2 == 0 else 0

        h, w = image.shape[:2]
        cxmin, cxmax = mask_size_half, w + offset - mask_size_half
        cymin, cymax = mask_size_half, h + offset - mask_size_half
        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin, ymin = cx - mask_size_half, cy - mask_size_half #左上角
        xmax, ymax = xmin + self.mask_size, ymin + self.mask_size #右下角
        xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax)
        if np.random.uniform(0, 1) < self.p:
            image[ymin:ymax, xmin:xmax] = (0, 0, 0)
            label[ymin:ymax, xmin:xmax] = 0
        return image, label

class Transform:
    def __init__(self):
        pass

    def crop_resize_data(self, image, label=None, image_size=(1024, 384), offset=690):
        """
        image.shape = h, w, c
        cv2.resize(image, (w, h))
        """
        roi_image = image[offset:, :] #上边offset个像素不要
        train_image = cv2.resize(roi_image, image_size, interpolation=cv2.INTER_LINEAR)

        if label is not None:
            roi_label = label[offset:, :]
            train_label = cv2.resize(roi_label, image_size, interpolation=cv2.INTER_NEAREST)
            return train_image, train_label

        return train_image

    def encode_labels(self, origin_label):
        encode_label = np.zeros((origin_label.shape[0], origin_label.shape[1]))
        id_train = {0:[0, 249, 255, 213, 206, 207, 211, 208, 216, 215, 218, 219, 232, 202, 231, 230, 228, 229, 233, 212, 223],
                    1:[200, 204, 209], 2: [201, 203], 3:[217], 4:[210], 5:[214],
                    6:[220, 221, 222, 224, 225, 226], 7:[205, 227, 250]}
        for i in range(8):
            for item in id_train[i]:
                encode_label[origin_label == item] = i

        return encode_label

    def __call__(self, image, label):
        # print(image.shape, label.shape)
        image, label = self.crop_resize_data(image, label)
        # print(image.shape, label.shape)
        label = self.encode_labels(label)
        # print(image.shape, label.shape)
        # sample = [image.copy(), label.copy()]

        # transforms = torchvision.transform.Compose([ImageAug(), DeformAug()])
        image, label = ImageAug()(image, label)
        image, label = DeformAug()(image, label)
        image, label = ScaleAug()(image, label)
        image, label = CutOut(32, 0.5)(image, label)
        image = np.transpose(image, (2, 0, 1))

        return torch.FloatTensor(image), torch.FloatTensor(label)

        # img_x = img_x / 127.0 - 1

        # label = np.zeros((9, img_label.shape[0], img_label.shape[1]))
        # label = np.stack([img_label] * 9)

        # label[0] = np.where((label[0] == 0) | (label[0] == 249) | (label[0] == 255), 1, 0)
        # label[1] = np.where((label[1] == 200) | (label[1] == 204) | (label[1] == 213) | (label[1] == 209) | (label[1] == 206) | (label[1] == 207), 1, 0)
        # label[2] = np.where((label[2] == 201) | (label[2] == 203) | (label[2] == 211) | (label[2] == 208), 1, 0)
        # label[3] = np.where((label[3] == 216) | (label[3] == 217) | (label[3] == 215), 1, 0)
        # label[4] = np.where((label[4] == 218) | (label[4] == 219), 1, 0)
        # label[5] = np.where((label[5] == 210) | (label[5] == 232), 1, 0)
        # label[6] = np.where(label[6] == 214, 1, 0)
        # label[7] = np.where((label[7] == 202)
        #                     | (label[7] == 220)
        #                     | (label[7] == 220)
        #                     | (label[7] == 221)
        #                     | (label[7] == 222)
        #                     | (label[7] == 231)
        #                     | (label[7] == 224)
        #                     | (label[7] == 225)
        #                     | (label[7] == 226)
        #                     | (label[7] == 230)
        #                     | (label[7] == 228)
        #                     | (label[7] == 229)
        #                     | (label[7] == 233), 1, 0)
        # label[8] = np.where((label[8] == 205) | (label[8] == 212) | (label[8] == 227) | (label[8] == 223) | (label[8] == 250), 1, 0)

        # for i in range(9):
        #     plt.figure()
        #     plt.imshow(label[i])
        #     plt.savefig('./label%d.jpg' % i)
        #     plt.close()

        # return torch.FloatTensor(img_x), torch.FloatTensor(label)