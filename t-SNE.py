#coding:utf-8
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from pylab import *
import os
import sys
import glob
import numpy as np
import mpl_layout

from tsne import bh_sne
from PIL import Image, ImageDraw
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.axes_grid.anchored_artists import AnchoredAuxTransformBox
#from matplotlib.patches import patches
from extract_img_feature import extract_feature


def get_attribute(choice, txtfilename='./out_attr.txt'):
    with open(txtfilename, 'r') as f:
        lines = f.readlines()
    Attr = []
    lines = lines[0:choice]
    for line in lines:
        line = line.split()
        attr = np.asarray(line, dtype=np.float32)
        Attr.append(attr)

    new_attr = np.asarray(Attr)
    attribute = new_attr[:, 20]
    return attribute

def vertical_stack_color_img(img_path_list, img_dir):
    for i, path in enumerate(img_path_list):
        img_path = os.path.join(img_dir, path)
        img = Image.open(img_path)
        img = np.ravel(img)
        try:
            img_list = np.vstack((img_list, img))
        except:
            img_list = img

    return img_list

def paste_image(path, coodinate, attr_num, ax=None, zoom=0.02):
    if ax is None:
        ax = plt.gca()
    x, y = coodinate
    if attr_num == 1:
        Attribute_color = (255, 0, 0)
    else:
        Attribute_color = (0, 0, 255)

    rec = Image.new('RGB', (212, 276), Attribute_color)
    img = Image.open(path)
    rec.paste(img, (10, 10))
    image_box = OffsetImage(rec, zoom=zoom)

    artists = []
    ab = AnnotationBbox(image_box, (x, y), xycoords='data', frameon=False)
    ax.add_artist(ab)
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    

def tsne_core(choice=50000, out_dir='results'):
    dir_name = './../../Deep_Learning_Datasets/CelebA_Datasets'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_path_list = os.listdir(dir_name)
    if img_path_list[0] == '.DS_Store':
        del img_path_list[0]
    img_path_list = img_path_list[0:choice]
    img_list = extract_feature(img_path_list)
    #img_list = vertical_stack_color_img(img_path_list=img_path_list, img_dir=dir_name)
    print('Image list shape : {}'.format(img_list.shape))
    X = img_list.astype(np.float64)

    data = bh_sne(data=X, d=2, perplexity=30., theta=0.5)
    
    x, y = data[:, 0], data[:, 1]
    
    fig, ax = plt.subplots()

    attributes = get_attribute(choice)
    #coodinates = data.tolist()
    for filename, coodinate, attribute in zip(img_path_list, data, attributes):
        paste_image(path=os.path.join(dir_name, filename), coodinate=coodinate, attr_num=attribute, ax=ax)

    ax.tick_params(labelbottom='off', bottom='off')
    ax.tick_params(labelleft='off', left='off')
    ax.set_xticklabels([])
    box('off')

    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(color='black')
    #plt.scatter(x, y)
    plt.savefig('t-SNE_t.jpg', dpi=2000)


if __name__ == '__main__':
    mpl_layout.layout()
    args = sys.argv
    tsne_core()
