import os
import pandas as pd

def gen_list():
    root = '/home/zhaochengshuai/dataset/cv/ColorImage'
    inputs = []
    zcs(root, inputs)
    print(len(inputs))

    labels = []
    for x in inputs:
        label = x.replace('ColorImage/ColorImage', 'Gray_Label/Label')
        label = label.replace('ColorImage', 'Label')
        label = label[:-4] + "_bin.png"
        labels.append(label)
        # print(label)
        # break
    # print(len(labels))
    assert len(labels) == len(inputs)
    return inputs, labels

def zcs(cur_dir, inputs):
    files = os.listdir(cur_dir)
    assert len(files) > 0
    if files[0].endswith('jpg'):
        inputs += [os.path.join(cur_dir, file) for file in files]
        # print(inputs)
    else:
        for file in files:
            zcs(os.path.join(cur_dir, file), inputs)

def check_list(inputs, lables):
    for x, label in zip(inputs, labels):
        assert os.path.exists(x)
        assert os.path.exists(label)

def save_list(inputs, labels):
    df = pd.DataFrame(data=zip(inputs, labels))
    df.to_csv("./dataset.csv", index=False)


if __name__ == "__main__":
    inputs, labels = gen_list()
    # check_list(inputs, labels)
    save_list(inputs, labels)