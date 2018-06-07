import argparse
import chainer
import numpy as np
import network
import os

from tqdm import tqdm
from PIL import Image
from chainer import serializers, Variable


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', '-n', type=int, default=100)
    parser.add_argument('--depth', '-d', type=int, default=5)
    parser.add_argument('--dis', type=str, default='./results/dis')
    parser.add_argument('--img', '-i', type=str, default='./../../Deep_Learning_Datasets/CelebA_Datasets')
    parser.add_argument('--txt', '-t', type=str, default='./out_attr.txt')
    args = parser.parse_args()

    return args

def make_attribute():
    args = parser()
    with open(args.txt, 'r') as f:
        lines = f.read()
    lines = lines.split()
    mat = np.asarray(lines, dtype=np.float32).reshape(202599, 40)
    use_attr = [20, 15, 31, 24, 35]
    return 0
    
def random_box(size, d):
     l = np.random.randint(0, d+1)
     t = np.random.randint(0, d+1)
     r = size[0] - (d-l)
     b = size[1] - (d-t)
     return (l, t, r, b)

def get_example(item, depth, i):

    img = Image.open(item)
    img = img.crop(random_box(img.size, 16))

    h = 2**(2+depth)
    w = 3*(2**depth)
    img = img.resize((w,h))
    img = np.asarray(img, dtype=np.float32) / 256.
    if len(img.shape) == 2:
        img = np.broadcast_to(img, (3, img.shape[0], img.shape[1]))
    else:
        img = np.transpose(img, (2, 0, 1))

    return img

def extract_feature(path):
    args = parser()

    dis = network.Discriminator(args.depth)
    print('Loading Discirminator Model from ' + args.dis)
    serializers.load_npz(args.dis, dis)    
    
    img_list = os.listdir(args.img)
    img_list = sorted(img_list)
    if img_list[0] == '.DS_Store':
        del img_list[0]

    IMG_PATH = [os.path.join(args.img, name) for name in img_list]
    for i in tqdm(range(args.num)):
        im = get_example(os.path.join(args.img, IMG_PATH[i]), args.depth, None)
        im = im[np.newaxis, :, :, :]
        out_dis, _ = dis(im, alpha=1.0)
        out_dis = np.ravel(out_dis.data)
        try:
            output = np.vstack((output, out_dis))
        except:
            output = out_dis

    return output
    

if __name__ == '__main__':
    output = extract_feature()
    print(output.shape)
