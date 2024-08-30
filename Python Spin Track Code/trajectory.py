import glob
import pickle
from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def trajectory(fl):
    particles = []
    with open(fl,'rb') as f:
        data = pickle.load(f)
    for di,d in tqdm(enumerate(data)):
        if di == 0:
            for o,c0,c1,fi in zip(d['orientation'],d['centroid-0'],d['centroid-1'],d['frame']):
                particles.append({'frame':[fi], 'orientation': [o],'centroid-0': [c0],'centroid-1': [c1]})
        else:
            index_match = []
            for p in particles:
                dsp = np.sqrt((d['centroid-0']-p['centroid-0'][-1])**2+(d['centroid-1']-p['centroid-1'][-1])**2)
                min_dsp=min(dsp)
                frame_gap = int(d['frame'][0]-p['frame'][-1])
                if min_dsp<23 and frame_gap<16.1:
                    idx = np.argmin(dsp)
                    p['frame'].append(d['frame'][idx])
                    p['orientation'].append(d['orientation'][idx])
                    p['centroid-0'].append(d['centroid-0'][idx])
                    p['centroid-1'].append(d['centroid-1'][idx])
                    index_match.append(idx)
            '''
            for indx,(o,c0,c1,fi) in enumerate(zip(d['orientation'],d['centroid-0'],d['centroid-1'],d['frame'])):
                if indx not in index_match:
                    particles.append({'frame':[fi], 'orientation': [o],'centroid-0': [c0],'centroid-1': [c1]})
            '''
    results = []
    for p in particles:
        results.append(pd.DataFrame(data=p))
    return results

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('video_name', help='video name. e.g.2024-06-05-cluster_6')
    args = parser.parse_args()
    fp= '../data/{}.pickle'.format(args.video_name)
    save_path = fp[:-7] + '-trj_all.pickle'
    results = trajectory(fp)
    with open(save_path, 'wb') as f:
        pickle.dump(results, f, -1)
if __name__=='__main__':
    main()

