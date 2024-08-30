import glob
import pickle
from pathlib import Path
import argparse
import os
import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm
from numpy import pi, cos, sin, sqrt, arctan2
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator,FixedLocator
from scipy.stats import circmean, circstd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.ndimage import rotate
from imageio import get_writer, get_reader
from matplotlib.patches import Circle
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes

def myfun(zdata, theta, x0, y0):
    # half=len(zdata)//2
    # print(zdata)
    # xvec = zdata[:half]
    # yvec = zdata[half:]
    # zdata = zdata.reshape((2,half))
    # x,y=zdata
    # input_data=np.zeros((len(x)))
    # input_data[0,:]=x[:]
    # input_data[1,:]=y[:]
    # print(zdata, zdata.shape)
    Rmat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    # input_data = np.vstack((xvec, yvec))
    # print(input_data)
    rotPos = Rmat @ (zdata - np.array([[x0], [y0]]))

    # return np.hstack((rotPos[0],rotPos[1]))
    # return rotPos.reshape(2*len(rotPos[0]))
    # print(rotPos.ravel().T,rotPos.ravel().T.shape)
    return rotPos.ravel().T

def cluster_rotation(min_fframe,start_frame,particles):
    # Load data (assuming you have already loaded posArr and mainClusterID)

    # Fit a function to relate x(t+1) and y(t+1) to x(t) and y(t)
    global_parameters = {'frame':[],'orientation':[],'centroid-0':[], 'centroid-1':[]}
    dt=1 #unit: frame
    for t in tqdm(range(start_frame, min_fframe - dt)):
        c0=[]
        c1=[]
        c0t=[]
        c1t=[]
        # print('frame:', t)
        for part in particles:
            frames=np.array(part['frame'])
            # print(frames,frames.shape)
            centroid0=np.array(part['centroid-0'])
            centroid1=np.array(part['centroid-1'])
            # print(centroid0,centroid0.shape)
            # print(centroid1,centroid1.shape)
            idx=np.where(abs(frames - t)<0.2)[0]
            if len(idx)==1:
                c0.append(centroid0[idx][0])
                c1.append(centroid1[idx][0])
            elif len(idx)==0:
                c0_interp=np.interp(t,frames,centroid0)
                c1_interp=np.interp(t,frames,centroid1)
                c0.append(c0_interp)
                c1.append(c1_interp)
            t1=t+dt
            idx1=np.where(abs(frames-t1)<0.2)[0]
            if len(idx1)==1:
                c0t.append(centroid0[idx1][0])
                c1t.append(centroid1[idx1][0])
            elif len(idx1)==0:
                c0_interp=np.interp(t1,frames,centroid0)
                c1_interp=np.interp(t1,frames,centroid1)
                c0t.append(c0_interp)
                c1t.append(c1_interp)
        # p0=[0.5,2,-2]
        # print(c0t)
        # print(c1t)
        t1_data=np.vstack((np.array(c0t).T,np.array(c1t).T))
        # print(t1_data, t1_data.shape)
        # print(c0+c1)
        t_data=np.array(c0+c1).ravel()
        # print(t_data, t_data.shape)
        p, _ = curve_fit(myfun, t1_data, t_data)
        global_parameters['frame'].append(t)
        global_parameters['orientation'].append(p[0])
        global_parameters['centroid-0'].append(p[1])
        global_parameters['centroid-1'].append(p[2])

        '''
        # Plot results (you can adjust this part as needed)
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.scatter(x1, y1, color='k')
        plt.scatter(x2, y2, color='r')
        plt.xlim(300, 1100)
        plt.ylim(100, 900)
        plt.box(True)

        f = myfun(p, [x2, y2])
        xvec, yvec = f[:len(f) // 2], f[len(f) // 2:]
        plt.subplot(122)
        plt.scatter(x1, y1, color='k')
        plt.scatter(xvec, yvec, color='b')
        plt.xlim(300, 1100)
        plt.ylim(100, 900)
        plt.title(f"Time step {t}")

        plt.pause(0.1)
        plt.clf()
        '''
    return global_parameters
def plot_results(particles, cluster,min_fframe,start_frame,video_name,video_Hz):

    # Load data
    part_num=len(particles)
    Nt = min_fframe-start_frame+1
    mat2p = np.full((Nt, part_num+1), np.nan)
    mat3p = np.full((Nt, part_num+1), np.nan)
    mat4p = np.full((Nt, part_num+1), np.nan)

    for i in range(part_num):
        vect = np.array(particles[i]['frame'])
        vec2 = np.array(particles[i]['orientation'])
        vec3 = np.array(particles[i]['centroid-0'])
        vec4 = np.array(particles[i]['centroid-1'])

        interp_func2 = interp1d(vect, vec2, fill_value="extrapolate")
        mat2p[:, i] = interp_func2(np.arange(start_frame, min_fframe+1))

        interp_func3 = interp1d(vect, vec3, fill_value="extrapolate")
        mat3p[:, i] = interp_func3(np.arange(start_frame, min_fframe+1))

        interp_func4 = interp1d(vect, vec4, fill_value="extrapolate")
        mat4p[:, i] = interp_func4(np.arange(start_frame, min_fframe+1))

    #cluster orientation and coordinates
    cvect = np.array(cluster['frame'])
    cvec2 = np.array(cluster['orientation'])
    cvec3 = np.array(cluster['centroid-0'])
    cvec4 = np.array(cluster['centroid-1'])

    interp_func2 = interp1d(cvect, cvec2, fill_value="extrapolate")
    cvec2_itv = interp_func2(np.arange(start_frame, min_fframe))
    interp_func3 = interp1d(cvect, cvec3, fill_value="extrapolate")
    cvec3_itv = interp_func3(np.arange(start_frame, min_fframe))
    interp_func4 = interp1d(cvect, cvec4, fill_value="extrapolate")
    cvec4_itv = interp_func4(np.arange(start_frame, min_fframe))
    # print(cvec2_itv, type(cvec2_itv), len(cvec2_itv))
    '''
    # Plot mat3p
    plt.figure()
    plt.plot(mat3p)
    plt.show()

    # Plot mat4p
    plt.figure()
    plt.plot(mat4p)
    plt.show()

    # Create animation
    fig, ax = plt.subplots()
    for t in range(1, 10001, 20):
        x = mat3p[t, :]
        y = mat4p[t, :]
        for (xi, yi) in zip(x, y):
            circle = Circle((xi, yi), 50, edgecolor='b', facecolor='none')
            ax.add_patch(circle)
        plt.xlim([-100, 780])
        plt.ylim([0, 700])
        plt.pause(0.1)
        plt.cla()

    # Plot mat2p
    plt.figure()
    plt.plot(mat2p)
    plt.show()
    '''
    # print(mat2p[:, part_num])

    mat2pp = np.zeros((Nt-1, part_num+1))
    window = np.bartlett(2*10+1)
    window2 = np.bartlett(2*60+1)
    # plt.figure()
    for i in range(part_num):

        vecTheta = mat2p[:, i]
        dvec = np.diff(vecTheta)

        idx = np.where(np.abs(dvec) > 1)[0]

        vect = np.arange(1, Nt)
        vect2 = np.delete(vect, idx)
        dvec2 = np.delete(dvec, idx)

        interp_func = interp1d(vect2, dvec2, fill_value="extrapolate")
        dvecp = interp_func(vect)
        # print(dvecp, type(dvecp), len(dvecp))
        # smoothed_dvecp = pd.Series(dvecp).rolling(window=20).mean().values
        smoothed_dvecp=np.convolve(dvecp, window2, mode='same')
        # plt.plot(smoothed_dvecp)
        mat2pp[:, i] = smoothed_dvecp
        # plt.pause(0.1)
        # plt.hold(True)
    # smoothed_dvecp = pd.Series(cvec2_itv).rolling(window=20).mean().values
    # print(smoothed_dvecp)
    # mat2pp[:, -1] = smoothed_dvecp

    cvec2_itv_sm = np.convolve(cvec2_itv, window, mode='same')
    mat2pp[:, -1] = cvec2_itv_sm


    # print(max(mat2pp[:, part_num]),min(mat2pp[:, part_num]))
    # Create video

    with get_writer('../results/{}_results_corotate.mp4'.format(video_name), fps=10, quality=9) as writer:
        lin_clr = plt.get_cmap('tab10', part_num+1)

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        plt.subplots_adjust(wspace=0.4)
        fig.patch.set_facecolor('white')

        for t in range(501, Nt-1000, 10):
            ax[1].clear()
            ax[1].plot(mat2pp[t-500:t+500, :part_num])
            ax[1].plot(mat2pp[t-500:t+500, -1], label = 'cluster', color='grey')
            ax[1].axvline(x=500, color='k', linestyle='--')
            ax[1].axhline(y=0, color='k')
            ax[1].set_xlim(0, 1000)
            # ax[1].set_ylim(-3.14, 3.14)
            ax[1].set_title(f'Time: {t}')
            ax[1].set_ylabel(r'$\omega$ (radian/frame)')
            ax[1].legend()
            # ax11=ax[1].twinx()
            # ax11.plot(mat2pp[t-500:t+500, -1])
            # ax11.set_ylim(-0.04,0.04)

            ax[0].clear()
            # theta_c = np.sum(mat2pp[t-10:t, -1])
            theta_c = np.sum(mat2pp[:t, -1])
            x=mat3p[t, :]
            y=mat4p[t, :]
            for i in range(part_num):
                circle = Circle((x[i], y[i]), 50, edgecolor=lin_clr(i), facecolor='none')
                ax[0].add_patch(circle)
                th = mat2p[t, i]
                l = 20
                xl = x[i] - l * np.cos(th)
                xr = x[i] + l * np.cos(th)
                yl = y[i] - l * np.sin(th)
                yr = y[i] + l * np.sin(th)
                ax[0].plot([xl, xr], [yl, yr], color=lin_clr(i), linewidth=2)
            ax[0].set_xlim([-100, 780])
            ax[0].set_ylim([0, 700])
            ax[0].set_aspect('equal')
            # ax[0].bbox(True)

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(*fig.canvas.get_width_height()[::-1],4)
            writer.append_data(image)
            ax[0].clear()
def plot_results_corotate(particles, cluster,min_fframe,start_frame,video_name,video_Hz):

    # Load data
    part_num=len(particles)
    Nt = min_fframe-start_frame+1
    mat2p = np.full((Nt, part_num+1), np.nan)
    mat3p = np.full((Nt, part_num+1), np.nan)
    mat4p = np.full((Nt, part_num+1), np.nan)

    for i in range(part_num):
        vect = np.array(particles[i]['frame'])
        vec2 = np.array(particles[i]['orientation'])
        vec3 = np.array(particles[i]['centroid-0'])
        vec4 = np.array(particles[i]['centroid-1'])

        interp_func2 = interp1d(vect, vec2, fill_value="extrapolate")
        mat2p[:, i] = interp_func2(np.arange(start_frame, min_fframe+1))

        interp_func3 = interp1d(vect, vec3, fill_value="extrapolate")
        mat3p[:, i] = interp_func3(np.arange(start_frame, min_fframe+1))

        interp_func4 = interp1d(vect, vec4, fill_value="extrapolate")
        mat4p[:, i] = interp_func4(np.arange(start_frame, min_fframe+1))

    #cluster orientation and coordinates
    cvect = np.array(cluster['frame'])
    cvec2 = np.array(cluster['orientation'])
    cvec3 = np.array(cluster['centroid-0'])
    cvec4 = np.array(cluster['centroid-1'])

    interp_func2 = interp1d(cvect, cvec2, fill_value="extrapolate")
    cvec2_itv = interp_func2(np.arange(start_frame, min_fframe))
    interp_func3 = interp1d(cvect, cvec3, fill_value="extrapolate")
    cvec3_itv = interp_func3(np.arange(start_frame, min_fframe))
    interp_func4 = interp1d(cvect, cvec4, fill_value="extrapolate")
    cvec4_itv = interp_func4(np.arange(start_frame, min_fframe))
    # print(cvec2_itv, type(cvec2_itv), len(cvec2_itv))
    '''
    # Plot mat3p
    plt.figure()
    plt.plot(mat3p)
    plt.show()

    # Plot mat4p
    plt.figure()
    plt.plot(mat4p)
    plt.show()

    # Create animation
    fig, ax = plt.subplots()
    for t in range(1, 10001, 20):
        x = mat3p[t, :]
        y = mat4p[t, :]
        for (xi, yi) in zip(x, y):
            circle = Circle((xi, yi), 50, edgecolor='b', facecolor='none')
            ax.add_patch(circle)
        plt.xlim([-100, 780])
        plt.ylim([0, 700])
        plt.pause(0.1)
        plt.cla()

    # Plot mat2p
    plt.figure()
    plt.plot(mat2p)
    plt.show()
    '''
    # print(mat2p[:, part_num])

    mat2pp = np.zeros((Nt-1, part_num+1))
    window = np.bartlett(2*10+1)
    window2 = np.bartlett(2*60+1)
    # plt.figure()
    for i in range(part_num):

        vecTheta = mat2p[:, i]
        dvec = np.diff(vecTheta)

        idx = np.where(np.abs(dvec) > 1)[0]

        vect = np.arange(1, Nt)
        vect2 = np.delete(vect, idx)
        dvec2 = np.delete(dvec, idx)

        interp_func = interp1d(vect2, dvec2, fill_value="extrapolate")
        dvecp = interp_func(vect)
        # print(dvecp, type(dvecp), len(dvecp))
        # smoothed_dvecp = pd.Series(dvecp).rolling(window=20).mean().values
        smoothed_dvecp=np.convolve(dvecp, window2, mode='same')
        # plt.plot(smoothed_dvecp)
        mat2pp[:, i] = smoothed_dvecp
        # plt.pause(0.1)
        # plt.hold(True)
    # smoothed_dvecp = pd.Series(cvec2_itv).rolling(window=20).mean().values
    # print(smoothed_dvecp)
    # mat2pp[:, -1] = smoothed_dvecp

    cvec2_itv_sm = np.convolve(cvec2_itv, window, mode='same')
    mat2pp[:, -1] = cvec2_itv_sm

    def setup_axes1(fig, rect, deg):
        """
        A simple one.
        """
        tr = Affine2D().rotate_deg(deg)

        grid_helper = floating_axes.GridHelperCurveLinear(
            tr, extremes=(-100, 780, 0, 880))
        #7,5 defalt for orthoganol plot


        ax1 = fig.add_subplot(
            rect, axes_class=floating_axes.FloatingAxes, grid_helper=grid_helper)
        # ax1.grid()
        p_deg=np.abs(deg)%45.0
        if p_deg==0.0:
            m_deg=0.0
        else:
            m_deg=45.0-p_deg
        # print(deg,m_deg,np.sqrt(2)*0.6*np.cos(np.deg2rad(m_deg)))
        left_margin=(1-np.sqrt(2)*0.6*np.cos(np.deg2rad(m_deg)))/2.0
        right_margin=1-left_margin
        bottom_margin=(1-np.sqrt(2)*0.6*np.cos(np.deg2rad(m_deg)))/2.0
        top_margin=1-left_margin
        # print(left_margin,right_margin)
        fig.subplots_adjust(left=left_margin, right=right_margin, bottom=bottom_margin, top=top_margin)

        ax1.axis["right"].set_visible(False)
        ax1.axis["top"].set_visible(False)
        ax1.axis["left"].set_visible(False)
        ax1.axis["bottom"].set_visible(False)
        ax1.set_aspect('equal', adjustable='box', anchor='C')
        aux_ax = ax1.get_aux_axes(tr)

        return ax1, aux_ax
    # print(max(mat2pp[:, part_num]),min(mat2pp[:, part_num]))
    # Create video

    with get_writer('../results/{}_results_corotate_test.mp4'.format(video_name), fps=10, quality=9) as writer:
        lin_clr = plt.get_cmap('tab10', part_num+1)

        fig= plt.figure(figsize=(8,8))
        # plt.subplots_adjust(wspace=0.4)
        fig.patch.set_facecolor('white')

        for t in range(501, Nt-1000, 10):
            if t==501:
                theta_c = 0
            else:
                theta_c = np.sum(cvec2_itv_sm[t-10:t])
            ax1, aux_ax1 = setup_axes1(fig, 111, np.rad2deg(-theta_c))
            x=mat3p[t, :]
            y=mat4p[t, :]
            for i in range(part_num):
                circle = Circle((x[i], y[i]), 50, edgecolor=lin_clr(i), facecolor='none')
                aux_ax1.add_patch(circle)
                th = mat2p[t, i]
                l = 20
                xl = x[i] - l * np.cos(th)
                xr = x[i] + l * np.cos(th)
                yl = y[i] - l * np.sin(th)
                yr = y[i] + l * np.sin(th)
                aux_ax1.plot([xl, xr], [yl, yr], color=lin_clr(i), linewidth=2)
            # ax[0].bbox(True)
            # aux_ax1.frame_on(False)
            # fig.tight_layout()
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(*fig.canvas.get_width_height()[::-1],4)
            writer.append_data(image)
            fig.clear(True)
def plot_results_corotate2(particles, cluster,min_fframe,start_frame,video_name,video_Hz):

    # Load data
    part_num=len(particles)
    Nt = min_fframe-start_frame+1
    mat2p = np.full((Nt, part_num+1), np.nan)
    mat3p = np.full((Nt, part_num+1), np.nan)
    mat4p = np.full((Nt, part_num+1), np.nan)

    for i in range(part_num):
        vect = np.array(particles[i]['frame'])
        vec2 = np.array(particles[i]['orientation'])
        vec3 = np.array(particles[i]['centroid-0'])
        vec4 = np.array(particles[i]['centroid-1'])

        interp_func2 = interp1d(vect, vec2, fill_value="extrapolate")
        mat2p[:, i] = interp_func2(np.arange(start_frame, min_fframe+1))

        interp_func3 = interp1d(vect, vec3, fill_value="extrapolate")
        mat3p[:, i] = interp_func3(np.arange(start_frame, min_fframe+1))

        interp_func4 = interp1d(vect, vec4, fill_value="extrapolate")
        mat4p[:, i] = interp_func4(np.arange(start_frame, min_fframe+1))

    #cluster orientation and coordinates
    cvect = np.array(cluster['frame'])
    cvec2 = np.array(cluster['orientation'])
    cvec3 = np.array(cluster['centroid-0'])
    cvec4 = np.array(cluster['centroid-1'])

    interp_func2 = interp1d(cvect, cvec2, fill_value="extrapolate")
    cvec2_itv = interp_func2(np.arange(start_frame, min_fframe))
    interp_func3 = interp1d(cvect, cvec3, fill_value="extrapolate")
    cvec3_itv = interp_func3(np.arange(start_frame, min_fframe))
    interp_func4 = interp1d(cvect, cvec4, fill_value="extrapolate")
    cvec4_itv = interp_func4(np.arange(start_frame, min_fframe))
    # print(cvec2_itv, type(cvec2_itv), len(cvec2_itv))
    '''
    # Plot mat3p
    plt.figure()
    plt.plot(mat3p)
    plt.show()

    # Plot mat4p
    plt.figure()
    plt.plot(mat4p)
    plt.show()

    # Create animation
    fig, ax = plt.subplots()
    for t in range(1, 10001, 20):
        x = mat3p[t, :]
        y = mat4p[t, :]
        for (xi, yi) in zip(x, y):
            circle = Circle((xi, yi), 50, edgecolor='b', facecolor='none')
            ax.add_patch(circle)
        plt.xlim([-100, 780])
        plt.ylim([0, 700])
        plt.pause(0.1)
        plt.cla()

    # Plot mat2p
    plt.figure()
    plt.plot(mat2p)
    plt.show()
    '''
    # print(mat2p[:, part_num])

    mat2pp = np.zeros((Nt-1, part_num+1))
    window = np.bartlett(2*10+1)
    window2 = np.bartlett(2*60+1)
    # plt.figure()
    for i in range(part_num):

        vecTheta = mat2p[:, i]
        dvec = np.diff(vecTheta)

        idx = np.where(np.abs(dvec) > 1)[0]

        vect = np.arange(1, Nt)
        vect2 = np.delete(vect, idx)
        dvec2 = np.delete(dvec, idx)

        interp_func = interp1d(vect2, dvec2, fill_value="extrapolate")
        dvecp = interp_func(vect)
        # print(dvecp, type(dvecp), len(dvecp))
        # smoothed_dvecp = pd.Series(dvecp).rolling(window=20).mean().values
        smoothed_dvecp=np.convolve(dvecp, window2, mode='same')
        # plt.plot(smoothed_dvecp)
        mat2pp[:, i] = smoothed_dvecp
        # plt.pause(0.1)
        # plt.hold(True)
    # smoothed_dvecp = pd.Series(cvec2_itv).rolling(window=20).mean().values
    # print(smoothed_dvecp)
    # mat2pp[:, -1] = smoothed_dvecp

    cvec2_itv_sm = np.convolve(cvec2_itv, window, mode='same')
    mat2pp[:, -1] = cvec2_itv_sm


    # print(max(mat2pp[:, part_num]),min(mat2pp[:, part_num]))
    # Create video

    with get_writer('../results/{}_results_corotate2_test.mp4'.format(video_name), fps=10, quality=9) as writer:
        lin_clr = plt.get_cmap('tab10', part_num+1)

        fig, ax = plt.subplots(figsize=(7, 5))
        plt.subplots_adjust(wspace=0.4)
        fig.patch.set_facecolor('white')

        for t in range(501, Nt-1000, 10):

            ax.clear()
            # theta_c = np.sum(mat2pp[t-10:t, -1])
            if t==501:
                theta_c = 0
            else:
                theta_c = np.sum(cvec2_itv_sm[t-10:t])
            x_intl=mat3p[t, :]
            y_intl=mat4p[t, :]
            x=x_intl*np.cos(-theta_c)-y_intl*np.sin(-theta_c)
            y=x_intl*np.sin(-theta_c)+y_intl*np.cos(-theta_c)
            for i in range(part_num):
                circle = Circle((x[i], y[i]), 50, edgecolor=lin_clr(i), facecolor='none')
                ax.add_patch(circle)
                th = mat2p[t, i]
                l = 20
                xl = x[i] - l * np.cos(th)
                xr = x[i] + l * np.cos(th)
                yl = y[i] - l * np.sin(th)
                yr = y[i] + l * np.sin(th)
                ax.plot([xl, xr], [yl, yr], color=lin_clr(i), linewidth=2)
            ax.set_xlim([-700, 700])
            ax.set_ylim([-50, 850])
            ax.set_aspect('equal')
            # ax[0].bbox(True)

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(*fig.canvas.get_width_height()[::-1],4)
            writer.append_data(image)
            ax.clear()
def plot_results_corotate3(particles, cluster,min_fframe,start_frame,video_name,video_Hz):

    # Load data
    part_num=len(particles)
    Nt = min_fframe-start_frame+1
    mat2p = np.full((Nt, part_num+1), np.nan)
    mat3p = np.full((Nt, part_num+1), np.nan)
    mat4p = np.full((Nt, part_num+1), np.nan)

    for i in range(part_num):
        vect = np.array(particles[i]['frame'])
        vec2 = np.array(particles[i]['orientation'])
        vec3 = np.array(particles[i]['centroid-0'])
        vec4 = np.array(particles[i]['centroid-1'])

        interp_func2 = interp1d(vect, vec2, fill_value="extrapolate")
        mat2p[:, i] = interp_func2(np.arange(start_frame, min_fframe+1))

        interp_func3 = interp1d(vect, vec3, fill_value="extrapolate")
        mat3p[:, i] = interp_func3(np.arange(start_frame, min_fframe+1))

        interp_func4 = interp1d(vect, vec4, fill_value="extrapolate")
        mat4p[:, i] = interp_func4(np.arange(start_frame, min_fframe+1))

    #cluster orientation and coordinates
    cvect = np.array(cluster['frame'])
    cvec2 = np.array(cluster['orientation'])
    cvec3 = np.array(cluster['centroid-0'])
    cvec4 = np.array(cluster['centroid-1'])

    interp_func2 = interp1d(cvect, cvec2, fill_value="extrapolate")
    cvec2_itv = interp_func2(np.arange(start_frame, min_fframe))
    interp_func3 = interp1d(cvect, cvec3, fill_value="extrapolate")
    cvec3_itv = interp_func3(np.arange(start_frame, min_fframe))
    interp_func4 = interp1d(cvect, cvec4, fill_value="extrapolate")
    cvec4_itv = interp_func4(np.arange(start_frame, min_fframe))
    # print(cvec2_itv, type(cvec2_itv), len(cvec2_itv))
    '''
    # Plot mat3p
    plt.figure()
    plt.plot(mat3p)
    plt.show()

    # Plot mat4p
    plt.figure()
    plt.plot(mat4p)
    plt.show()

    # Create animation
    fig, ax = plt.subplots()
    for t in range(1, 10001, 20):
        x = mat3p[t, :]
        y = mat4p[t, :]
        for (xi, yi) in zip(x, y):
            circle = Circle((xi, yi), 50, edgecolor='b', facecolor='none')
            ax.add_patch(circle)
        plt.xlim([-100, 780])
        plt.ylim([0, 700])
        plt.pause(0.1)
        plt.cla()

    # Plot mat2p
    plt.figure()
    plt.plot(mat2p)
    plt.show()
    '''
    # print(mat2p[:, part_num])

    mat2pp = np.zeros((Nt-1, part_num+1))
    window = np.bartlett(2*10+1)
    window2 = np.bartlett(2*60+1)
    # plt.figure()
    for i in range(part_num):

        vecTheta = mat2p[:, i]
        dvec = np.diff(vecTheta)

        idx = np.where(np.abs(dvec) > 1)[0]

        vect = np.arange(1, Nt)
        vect2 = np.delete(vect, idx)
        dvec2 = np.delete(dvec, idx)

        interp_func = interp1d(vect2, dvec2, fill_value="extrapolate")
        dvecp = interp_func(vect)
        # print(dvecp, type(dvecp), len(dvecp))
        # smoothed_dvecp = pd.Series(dvecp).rolling(window=20).mean().values
        smoothed_dvecp=np.convolve(dvecp, window2, mode='same')
        # plt.plot(smoothed_dvecp)
        mat2pp[:, i] = smoothed_dvecp
        # plt.pause(0.1)
        # plt.hold(True)
    # smoothed_dvecp = pd.Series(cvec2_itv).rolling(window=20).mean().values
    # print(smoothed_dvecp)
    # mat2pp[:, -1] = smoothed_dvecp

    cvec2_itv_sm = np.convolve(cvec2_itv, window, mode='same')
    mat2pp[:, -1] = cvec2_itv_sm


    # print(max(mat2pp[:, part_num]),min(mat2pp[:, part_num]))
    # Create video

    with get_writer('../results/{}_results_corotate3_test.mp4'.format(video_name), fps=10, quality=9) as writer:
        lin_clr = plt.get_cmap('tab10', part_num+1)

        fig, ax = plt.subplots(figsize=(7, 5))
        plt.subplots_adjust(wspace=0.4)
        fig.patch.set_facecolor('white')

        for t in range(501, Nt-1000, 10):

            ax.clear()
            # theta_c = np.sum(mat2pp[t-10:t, -1])
            if t==501:
                theta_c = 0
            else:
                theta_c = np.sum(cvec2_itv[t-10:t])
            x_intl=mat3p[t, :]-(np.sum(mat3p[t, :])/part_num)
            y_intl=mat4p[t, :]-(np.sum(mat4p[t, :])/part_num)
            x=x_intl*np.cos(-theta_c)-y_intl*np.sin(-theta_c)
            y=x_intl*np.sin(-theta_c)+y_intl*np.cos(-theta_c)
            for i in range(part_num):
                circle = Circle((x[i], y[i]), 50, edgecolor=lin_clr(i), facecolor='none')
                ax.add_patch(circle)
                th = mat2p[t, i]
                l = 20
                xl = x[i] - l * np.cos(th)
                xr = x[i] + l * np.cos(th)
                yl = y[i] - l * np.sin(th)
                yr = y[i] + l * np.sin(th)
                ax.plot([xl, xr], [yl, yr], color=lin_clr(i), linewidth=2)
            ax.set_xlim([-700, 700])
            ax.set_ylim([-800, 800])
            ax.set_aspect('equal')
            # ax[0].bbox(True)

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(*fig.canvas.get_width_height()[::-1],4)
            writer.append_data(image)
            ax.clear()
def raw_video_rotate(particles, cluster,min_fframe,start_frame,video_name,video_Hz):
    vid = get_reader('../{}.mp4'.format(video_name),  'ffmpeg')
    Nt = min_fframe-start_frame+1
    #cluster orientation and coordinates
    cvect = np.array(cluster['frame'])
    cvec2 = np.array(cluster['orientation'])
    cvec3 = np.array(cluster['centroid-0'])
    cvec4 = np.array(cluster['centroid-1'])

    interp_func2 = interp1d(cvect, cvec2, fill_value="extrapolate")
    cvec2_itv = interp_func2(np.arange(start_frame, min_fframe))
    interp_func3 = interp1d(cvect, cvec3, fill_value="extrapolate")
    cvec3_itv = interp_func3(np.arange(start_frame, min_fframe))
    interp_func4 = interp1d(cvect, cvec4, fill_value="extrapolate")
    cvec4_itv = interp_func4(np.arange(start_frame, min_fframe))

    window = np.bartlett(2*10+1)

    cvec2_itv_sm = np.convolve(cvec2_itv, window, mode='same')


    # print(max(mat2pp[:, part_num]),min(mat2pp[:, part_num]))
    # Create video

    with get_writer('../results/{}_raw_video_corotate_test.mp4'.format(video_name), fps=10, quality=9) as writer:
        for t in range(501, Nt-1000, 10):
            # theta_c = np.sum(mat2pp[t-10:t, -1])
            if t==501:
                theta_c = 0
            else:
                theta_c = np.sum(cvec2_itv_sm[t-10:t])
            image = vid.get_data(t+start_frame)[100:808,:]
            rotated_image = rotate(image, np.rad2deg(-theta_c), reshape=False)
            writer.append_data(rotated_image)
def corrl_cluster(particles,cluster,min_fframe,start_frame,video_name):
    # Load data
    part_num=len(particles)
    Nt = min_fframe-start_frame+1
    mat_ort = np.full((Nt, part_num), np.nan)
    mat_c0 = np.full((Nt, part_num), np.nan)
    mat_c1 = np.full((Nt, part_num), np.nan)

    for i in range(part_num):
        vect = np.array(particles[i]['frame'])
        vec2 = np.array(particles[i]['orientation'])
        vec3 = np.array(particles[i]['centroid-0'])
        vec4 = np.array(particles[i]['centroid-1'])

        interp_func2 = interp1d(vect, vec2, fill_value="extrapolate")
        mat_ort[:, i] = interp_func2(np.arange(start_frame, min_fframe+1))

        interp_func3 = interp1d(vect, vec3, fill_value="extrapolate")
        mat_c0[:, i] = interp_func3(np.arange(start_frame, min_fframe+1))

        interp_func4 = interp1d(vect, vec4, fill_value="extrapolate")
        mat_c1[:, i] = interp_func4(np.arange(start_frame, min_fframe+1))

    #cluster orientation and coordinates
    cvect = np.array(cluster['frame'])
    cvec2 = np.array(cluster['orientation'])
    cvec3 = np.array(cluster['centroid-0'])
    cvec4 = np.array(cluster['centroid-1'])

    interp_func2 = interp1d(cvect, cvec2, fill_value="extrapolate")
    cvec2_itv = interp_func2(np.arange(start_frame, min_fframe))
    mat2_der = np.zeros((Nt-1, part_num+1))
    window = np.bartlett(2*10+1)
    for i in range(part_num):

        vecTheta = mat_ort[:, i]
        dvec = np.diff(vecTheta)

        idx = np.where(np.abs(dvec) > 1)[0]

        vect = np.arange(1, Nt)
        vect2 = np.delete(vect, idx)
        dvec2 = np.delete(dvec, idx)

        interp_func = interp1d(vect2, dvec2, fill_value="extrapolate")
        dvecp = interp_func(vect)
        # print(dvecp, type(dvecp), len(dvecp))
        # smoothed_dvecp = pd.Series(dvecp).rolling(window=20).mean().values
        smoothed_dvecp=np.convolve(dvecp, window, mode='same')

        mat2_der[:, i] = smoothed_dvecp

    # window = np.bartlett(2*10+1)
    cvec2_itv_sm = np.convolve(cvec2_itv, window, mode='same')
    mat2_der[:, -1] = cvec2_itv_sm
    min_cluster=min(mat2_der[:, -1])
    max_cluster=max(mat2_der[:, -1])
    print(min_cluster,max_cluster)
    bin_n=9
    f_range,step=np.linspace(min_cluster,max_cluster,num=bin_n,retstep=True)
    save=np.zeros((bin_n-1,2*part_num+1))
    save[:,0]=f_range[:-1]+step/2.0
    for ib, b in enumerate(f_range[:-1]):
        ind1=np.where(mat2_der[:, -1]>b)[0]
        ind2=np.where(mat2_der[:, -1]<=f_range[ib+1])[0]
        index=np.intersect1d(ind1,ind2)
        print(len(index))
        for im in range(part_num):
            save[ib,im+1]=np.mean(mat2_der[:,im][index])
            save[ib,im+1+part_num]=np.std(mat2_der[:,im][index])
    # print(save)
    fig,ax=plt.subplots()
    lines=[]
    for ip in range(part_num):
        line,=ax.plot(save[:,0],save[:,ip+1],'-o',label='particle_{}'.format(ip+1))
        lines += [line]
    leg = ax.legend(fancybox=True, shadow=True)
    lined = {}
    #print(lines)
    for legline, origline in zip(leg.get_lines(), lines):
        legline.set_picker(5)  # Enable picking on the legend line.
        lined[legline] = origline
    def on_pick(event):
        # On the pick event, find the original line corresponding to the legend
        # proxy line, and toggle its visibility.
        legline = event.artist
        origline = lined[legline]
        visible = not origline.get_visible()
        origline.set_visible(visible)
        # Change the alpha on the line in the legend, so we can see what lines
        # have been toggled.
        legline.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw()
    ax.set_xlabel(r'$\omega$ cluster')
    ax.set_ylabel(r'$\omega$ individual')
    ax.set_ylim(-1.5,0.3)
    ax.set_xlim(-0.14,0.03)
    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()
    fig.savefig('../results/{}-corrl_cluster_indv.png'.format(video_name),dpi=300)
    plt.close()
    np.savetxt('../results/{}-corrl_cluster_indv.dat'.format(video_name), save, header='cluster omega, omega of {} particles, std of omega of {} particles'.format(part_num,part_num))

def main():
    video_Hz=29.79
    parser=argparse.ArgumentParser()
    parser.add_argument('video_name', help='video name. e.g.2024-06-05-cluster_6')
    parser.add_argument('--cluster', action='store_true', help='calculate the cluster dynamics')
    parser.add_argument('--plot_all', action = 'store_true', help='plot all results include cluster')
    parser.add_argument('--corrl_cluster', action = 'store_true', help='correlation between cluster dynamics and individual spinners')
    args = parser.parse_args()
    fp = '../data/{}-trj_all.pickle'.format(args.video_name)
    with open(fp, 'rb') as f:
        particles = pickle.load(f)
    part0=particles[0]
    # print(np.array(part0['frame']), part0['frame'].shape)
    frame_part0=np.array(part0['frame'])
    min_fframe = frame_part0[-1]
    # print(min_fframe)
    start_frame= frame_part0[0]
    for part in particles[1:]:
        frame=np.array(part['frame'])
        if min_fframe > frame[-1]:
            min_fframe = frame[-1]
    min_fframe = int(min_fframe)
    start_frame = int(start_frame)
    if args.cluster:
        global_parameters=cluster_rotation(min_fframe,start_frame,particles)
        save_path='../data/{}-cluster.pickle'.format(args.video_name)
        with open (save_path, 'wb') as fs:
            pickle.dump(global_parameters,fs,-1)
    if args.plot_all:
        print(args.video_name)
        cluster_p='../data/{}-cluster.pickle'.format(args.video_name)
        with open(cluster_p,'rb') as fc:
            cluster=pickle.load(fc)
        # plot_results(particles,cluster,min_fframe,start_frame,args.video_name,video_Hz)
        # plot_results_corotate(particles,cluster,min_fframe,start_frame,args.video_name,video_Hz)
        plot_results_corotate3(particles,cluster,min_fframe,start_frame,args.video_name,video_Hz)
        # raw_video_rotate(particles,cluster,min_fframe,start_frame,args.video_name,video_Hz)
    if args.corrl_cluster:
        startf={'2024-06-05-cluster_6':871,'2024-06-04-cluster_5':2331,'2024-06-04-cluster_4':1731,'2024-06-04-cluster_3':871}
        start_frame_c=startf[args.video_name]+start_frame
        print(args.video_name)
        cluster_p='../data/{}-cluster.pickle'.format(args.video_name)
        with open(cluster_p,'rb') as fc:
            cluster=pickle.load(fc)
        corrl_cluster(particles,cluster,min_fframe,start_frame_c,args.video_name)


if __name__=='__main__':
    main()
