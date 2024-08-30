import glob
import argparse
import os
import pickle
from pathlib import Path
import imageio
## common numerical and scientific libraries
import numpy as np
from numpy import sqrt, pi, sin, cos
import pandas as pd
import skimage
import skimage.io
import skimage.segmentation
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
## other common libraries
from tqdm import tqdm

def image_proc(image, num):
    # print(image.shape)
    g_image=rgb2gray(image)[100:808,:]
    # print(g_image.shape)
    #from skimage.filters import try_all_threshold
    #fig, ax = try_all_threshold(g_image, figsize=(10, 8), verbose=False)
    # print(g_image)
    g_image= skimage.filters.median(g_image)
    g_image= skimage.filters.gaussian(g_image, 0.3)
    b_gimage = g_image<150/255
    #imb = skimage.morphology.flood_fill(g_image, (0,0))
    # plt.imshow(b_gimage)
    # plt.show()
    # plt.close()
    #g_image = img_as_ubyte(g_image)
    #skimage.io.imsave('g_example.png',g_image)
    labels = skimage.measure.label(b_gimage)
    stats = skimage.measure.regionprops_table(labels, properties= ['area','solidity','perimeter','major_axis_length','label','eccentricity'])
    indx1 = np.logical_and(stats['area']>35, stats['solidity']>0.87)
    #indx2 = np.logical_and(stats['perimeter']>60, stats['perimeter']<100)
    indx3 = np.logical_and(stats['major_axis_length']<20, stats['major_axis_length']>9)
    indx4 = np.logical_and(stats['eccentricity']>0.5,stats['eccentricity']<1.0)

    indx = np.logical_and(indx1,indx3)
    #indx = np.logical_and(indx,indx3)
    indx = np.logical_and(indx,indx4)
    #print(stats['eccentricity'][indx], min(stats['eccentricity'][indx]), np.mean(stats['eccentricity'][indx]))
    #print(stats['solidity'][indx], min(stats['solidity'][indx]),np.mean(stats['solidity'][indx]))
    #print(sum(indx))
    #print(stats['label'], len(stats['label']))
    for idx,label in zip(indx, stats['label']):
        if idx==False:
            labels[labels==label] = 0
    BW2 = labels>0
    # skimage.io.imshow(img_as_ubyte(BW2))
    # skimage.io.show()
    labels2 = skimage.measure.label(BW2)
    stats2 = skimage.measure.regionprops_table(labels2, properties = ['centroid', 'orientation'])
    stats2['frame'] = np.zeros(len(stats2['orientation']))+num
    #print(stats2.keys())
    save_pickle = pd.DataFrame(stats2)
    return save_pickle

def main():
    path='/mnt/c/Users/Bo Fan/Documents/Myfiles/Projects/UCSD-projects/2024-06-05-spinning_robots'

    parser=argparse.ArgumentParser()
    parser.add_argument('video_name', help='video name. e.g.2024-06-05-cluster_6')
    args = parser.parse_args()
    filename = '{}.mp4'.format(args.video_name)
    start_f= {'2024-06-05-cluster_6': 950,'2024-06-04-cluster_5':570 ,'2024-06-04-cluster_4':15,'2024-06-04-cluster_3':2250}
    vid = imageio.get_reader('{}/{}.mp4'.format(path,filename),  'ffmpeg')
    # number of frames in video
    num_frames=vid.count_frames()
    print(filename)
    print('total frames:', num_frames)
    save_allpickle = []
    save_path = '../data/{}.pickle'.format(filename.split('.')[0])
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for num in tqdm(range(start_f[args.video_name],num_frames)):
            image = vid.get_data(num)
            save_pickle = image_proc(image, num)
            save_allpickle.append(save_pickle)
    except IndexError:
        print('index error')

    with open(str(save_path), 'wb') as f:
        pickle.dump(save_allpickle, f, -1)

def test():
    path='..'
    filename = '2024-06-04-cluster_3.mp4'
    vid = imageio.get_reader('{}/{}'.format(path,filename),  'ffmpeg')
    # number of frames in video
    num_frames=vid._meta['nframes']
    num = 2250
    image = vid.get_data(num)
    print(image.shape)

    skimage.io.imshow(img_as_ubyte(image))
    skimage.io.show()
    g_image=rgb2gray(image)[100:808,:]
    print(g_image.shape)
    #from skimage.filters import try_all_threshold
    #fig, ax = try_all_threshold(g_image, figsize=(10, 8), verbose=False)
    print(g_image)
    g_image= skimage.filters.median(g_image)
    g_image= skimage.filters.gaussian(g_image, 0.3)
    b_gimage = g_image<150/255
    #imb = skimage.morphology.flood_fill(g_image, (0,0))
    plt.imshow(b_gimage)
    plt.show()
    plt.close()
    #g_image = img_as_ubyte(g_image)
    #skimage.io.imsave('g_example.png',g_image)
    labels = skimage.measure.label(b_gimage)
    stats = skimage.measure.regionprops_table(labels, properties= ['area','solidity','perimeter','major_axis_length','label','eccentricity'])
    indx1 = np.logical_and(stats['area']>35, stats['solidity']>0.87)
    #indx2 = np.logical_and(stats['perimeter']>60, stats['perimeter']<100)
    indx3 = np.logical_and(stats['major_axis_length']<20, stats['major_axis_length']>9)
    indx4 = np.logical_and(stats['eccentricity']>0.5,stats['eccentricity']<1.0)

    indx = np.logical_and(indx1,indx3)
    #indx = np.logical_and(indx,indx3)
    indx = np.logical_and(indx,indx4)
    #print(stats['eccentricity'][indx], min(stats['eccentricity'][indx]), np.mean(stats['eccentricity'][indx]))
    #print(stats['solidity'][indx], min(stats['solidity'][indx]),np.mean(stats['solidity'][indx]))
    #print(sum(indx))
    #print(stats['label'], len(stats['label']))
    for idx,label in zip(indx, stats['label']):
        if idx==False:
            labels[labels==label] = 0
    BW2 = labels>0
    skimage.io.imshow(img_as_ubyte(BW2))
    skimage.io.show()
if __name__=='__main__':
    # main()
    test()
