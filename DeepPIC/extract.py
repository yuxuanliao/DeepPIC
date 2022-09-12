# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 13:04:15 2022

@author: yxliao
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pyopenms import MSExperiment, MzXMLFile, MzMLFile, MzDataFile
from sklearn import preprocessing
from multiprocess import Pool
from DeepPIC.train import *
from DeepPIC.predict import *


def DeepPIC_load(saved_path):
    input_data = Input(shape=(256, 256, 1))
    model = get_unet(input_data, n_filters=64, dropout=0.5, batchnorm=True, padding='same')
    model.compile(optimizer=Adam(lr = 0.001), loss="binary_crossentropy", metrics=["accuracy"])
    model.load_weights(saved_path)
    return model


def readms(file_path):
    """
    Read mzXML, mzML and mzData files.
    Arguments:
        file_path: string
            path to the dataset locally
    Returns:
        Tuple of Numpy arrays: (m/z, intensity, retention time, mean interval of retention time).
    
    Examples:
        >>> from DeepPIC import readms
        >>> choose_spec0, rt, rt_mean_interval = readms("MM48.mzxml")
    """
    ms_format = os.path.splitext(file_path)[1]
    ms_format = ms_format.lower()
    msdata = MSExperiment()
    if ms_format == '.mzxml':
        file = MzXMLFile()
    elif ms_format == '.mzml':
        file = MzMLFile()
    elif ms_format == '.mzdata':
        file = MzDataFile()
    file.load(r'%s' % file_path, msdata)
    m_s = []
    intensity = []
    r_t = []
    rt = []
    for spectrum in msdata:
        if spectrum.getMSLevel() == 1:
            rt.append(spectrum.getRT())
            p_rt = []
            p_ms = []
            p_intensity = []
            rt1 = []
            for peak in spectrum:
                if peak.getIntensity() != 0:
                    p_rt.append(spectrum.getRT())
                    p_ms.append(peak.getMZ())
                    p_intensity.append(peak.getIntensity())   
            ms_index = np.argsort(np.positive(p_ms)) 
            r_t.extend(np.array(p_rt)[ms_index])              
            m_s.extend(np.array(p_ms)[ms_index])
            intensity.extend(np.array(p_intensity)[ms_index])
    rt2 = np.array(rt)
    rt1 = np.array(r_t)
    ms1 = np.array(m_s)
    intensity1 = np.array(intensity)
    choose_spec0 = np.c_[rt1,ms1,intensity1]
    rt = np.array(rt)
    if rt2.shape[0] > 1:
        rt_mean_interval = np.mean(np.diff(rt2))
    else:
        rt_mean_interval = 0.0 
    return choose_spec0, rt, rt_mean_interval


def get_range(choose_spec0, rt, rt_mean_interval, mass_inv = 0.08, rt_inv = 30, min_intensity=50000): 
    scan = len(rt)
    rt_inv = int(rt_inv/rt_mean_interval)
    c = np.ones(len(choose_spec0))
    tol_array = np.hstack(((choose_spec0),(c.reshape(len(c),1))))
    h_rt = []
    h_ms = []
    h_intensity = []
    choose_spec = []
    while True:
        tol_index = np.where(tol_array[:, 3] == 1)
        inded = (tol_array[tol_index[0], 2]).T
        max_ind = np.argmax(inded)
        max_intensity_intensity =inded[max_ind]
        ind = tol_index[0][max_ind]
        h_intensity_rt = tol_array[ind, 0]
        max_intensity_rt_i = np.searchsorted(rt,h_intensity_rt)
        max_int_ms = tol_array[ind, 1]
        if max_intensity_intensity < min_intensity:
            break
        else:    
            start = max_intensity_rt_i-rt_inv
            end = max_intensity_rt_i+rt_inv
            if start < 0:
                start = 0
            if end > scan:
                end = scan
            choose_rt = rt[start:end]
            ind_z = []
            for rti in range(len(choose_rt)):
                f = np.searchsorted(tol_array[:, 0],choose_rt[rti])
                h = np.searchsorted(tol_array[:, 0],choose_rt[rti],side='right')
                ind_f = np.searchsorted(tol_array[f:h,1],tol_array[f:h,1][np.abs(tol_array[f:h,1]-max_int_ms) < mass_inv])
                ind_t = ind_f+f
                ind_z.extend(ind_t)   
            index = np.array(ind_z)
            tol_array[index, 3] = 2
            rt1 = tol_array[index, 0]
            ms1 =  tol_array[index, 1]
            int1 = tol_array[index, 2]
            choose_spec_i = np.hstack(((rt1.reshape(len(rt1),1)),(ms1.reshape(len(ms1),1)),(int1.reshape(len(int1),1))))
            h_rt.append(h_intensity_rt)
            choose_spec.append(choose_spec_i)  
            h_intensity.append(max_intensity_intensity)
            h_ms.append(max_int_ms)
                           
    spec_rt = list(zip(choose_spec,h_rt,h_ms,h_intensity))
    #         np.savetxt('%s/%s_%s_%s_%s_%s_%s.txt' % ("D:/Dpic/data2/1167/train2",h_intensity_rt,max_int_ms,max_intensity_intensity,choose_spec_i[0,0],choose_spec_i[-1,0],choose_spec_i.shape[0]),choose_spec_i)
    return spec_rt

def get_array(choose_spec_r,np=np):
    array = []
    rt_b = []
    ms_b = []
    ms_range = []
    B = np.zeros([256, 256],dtype=float)
    ms_array = np.zeros([256, 256],dtype=float)
    N = np.unique(choose_spec_r[0][:,0])
    if ((256-len(N)) % 2) == 0:
        a = np.array(0).repeat(int((256-len(N))/2))
        b = np.array(0).repeat(int((256-len(N))/2))
    else:
        a = np.array(0).repeat(int((256-len(N))/2))
        b = np.array(0).repeat(int((256-len(N))/2)+1)
    left = np.hstack((a,N,b))
    g1 = np.searchsorted(choose_spec_r[0][:,0], choose_spec_r[1])
    g2 = np.searchsorted(choose_spec_r[0][:,0], choose_spec_r[1], side='right')
    g = np.searchsorted(choose_spec_r[0][g1:g2,1], choose_spec_r[0][g1:g2,1][choose_spec_r[0][g1:g2,1]==choose_spec_r[2]])
    C = np.around(choose_spec_r[0][g1+g,1], 3)
    k1 = np.around(np.arange(C-0.01,C+0.01,0.005), 3)
    k2 = np.around(np.arange(C-1.28,C+1.28,0.01), 3)
    k3 = np.delete(np.unique(np.hstack((k1,k2))),np.searchsorted(np.unique(np.hstack((k1,k2))), C))
    rows = dict(zip(list(range(0, len(B))), list(left)))
    cols = dict(zip(list(range(0, len(B)+1)), list(k3)))
    for row in range(256):
        for i in range(len(choose_spec_r[0])):
            if rows[row] == choose_spec_r[0][i,0]:
                for col in range(256):
                    if cols[col]<choose_spec_r[0][i,1]<=cols[col+1]:
                        B[col][row] = choose_spec_r[0][i,2]
                        ms_array[col][row] = choose_spec_r[0][i,1]
    rt_b.append(left)
    array.append(B)
    ms_b.append(ms_array)
    ms_range.append(k3)
    return array,rt_b,ms_b,ms_range  

def scaler(array):
    MM48_test = []
    for i in range(len(array)):
        min_max_scaler = preprocessing.MinMaxScaler()
        arl = min_max_scaler.fit_transform(array[i][0][0])
        MM48_test.append(arl)
    MM48_test_20 = np.expand_dims(np.array(MM48_test), axis=3)
    return MM48_test_20

def get_pred_array(preds):
    preds_array = []
    for pred in preds:
        preds_array.append(pred)
        for i in range(len(pred[:,:,0])):
            for j in range(len(pred[:,:,0])):
                if pred[i,j,0] <= 0.01:
                    pred[i,j,0] = 0
                else:
                    pred[i,j,0] = 1
    return preds_array

def pred_array(a,preds,array):
    choose_spec2 = []
    preds_array = get_pred_array(preds)
    for iu in range(len(array)):
        choose_spec11 = array[iu][a][0]
        for i in range(256):
            for j in range(256):
                if preds_array[iu][i,j,0] == 0.0:
                    choose_spec11[i,j] = 0.0 
        choose_spec1 = choose_spec11[126:131,:] 
        for i in range(256):
            if preds_array[iu][128,i,0] == 0.0:
                if preds_array[iu][126,i,0] == 1.0:
                    preds_array[iu][128,i,0] = preds_array[iu][126,i,0]
                if preds_array[iu][127,i,0] == 1.0:
                    preds_array[iu][128,i,0] = preds_array[iu][127,i,0]
                if preds_array[iu][129,i,0] == 1.0:
                    preds_array[iu][128,i,0] = preds_array[iu][129,i,0]
                if preds_array[iu][130,i,0] == 1.0:
                    preds_array[iu][128,i,0] = preds_array[iu][130,i,0]
        for i in range(256):
            if preds_array[iu][128,i,0] == 1 and choose_spec1[2,i] == 0.0:
                if len(choose_spec1[:,i][choose_spec1[:,i].nonzero()]) == 1:
                    choose_spec1[2,i] = choose_spec1[:,i][choose_spec1[:,i].nonzero()]
        for q1 in range(256):
            if preds_array[iu][128,q1] == 1:
                break
            # else:
            #     q1 = 128
        for q3 in range(256):
            if preds_array[iu][128,q3] == 1:
                q2 = q3
            # else:
            #     q2 = 127
        choose_spec1[2,0:q1] = 0
        choose_spec1[2,q2+1:] = 0
        choose_spec2.append(choose_spec1)
    return choose_spec2

def pics(array,pred_array_int,pred_array_mz,choose_spec):
    pic_rt = []
    pic_mz = []
    pic_intensity = []
    pic_index = []
    pics = []
    peaks = []
    for index in range(len(array)):
        N = array[index][1][0]
        for x in range(len(N[:])):
            if N[x] != 0:
                break
        for z in range(len(N[:])):
            if N[z] != 0:
                v = z   
        mz_1 = pred_array_mz[index][2,x:v+1]
        int_1 = pred_array_int[index][2,x:v+1]
        if int_1.any() == 0:
            continue
        else:
            for p1 in range(len(int_1)):
                if int_1[p1] != 0:
                    break
            for p3 in range(len(int_1)):
                if int_1[p3] != 0:
                    p2 = p3
            rt_2 = N[x:v+1][p1:p2+1]
            int_2 = int_1[p1:p2+1]
            mz_2 = mz_1[p1:p2+1]
            max_int_index = np.searchsorted(rt_2, choose_spec[index][1])
            if len(int_2)<=max_int_index or int_2[max_int_index]==0:
                continue
            else:
                pic_rt.append(rt_2)
                pic_intensity.append(int_2)
                pic_mz.append(mz_2)
                pic_index.append(max_int_index)
                pic_1 = np.transpose(np.array([rt_2, int_2, mz_2]))
                  
                for t1 in range(max_int_index):
                    if pic_1[:,1][abs(t1-max_int_index)] == 0:
                        break
                    else:
                        t1 = max_int_index+1
                for t2 in range(len(pic_1)-max_int_index):
                    if pic_1[:,1][abs(t2+max_int_index)] == 0:
                        break
                    else:
                        t2 = len(pic_1)-max_int_index
                pic_2 = np.delete(pic_1,range((t2+max_int_index),len(pic_1)),axis=0)
                pic_3 = np.delete(pic_2,range(0,(max_int_index-t1+1)),axis=0) 
                pics.append(pic_3)
    # for i in range(len(pics)):
    #     np.savetxt('%s/%s.txt' % ('D:/Dpic/data2/leaf_seed/pics20',i), pics[i])  
    # np.savetxt('D:/Dpic/data2/leaf_seed/scantime20/rt20.txt', rt)
    return pics

def choose_label2(path,choose_spec_r0,array):
    files= os.listdir(path)
    choose_label = []
    label_path = []
    for file in files:
        position = path+'\\'+ file
        label_path.append(position)
        choose_label_i = np.loadtxt(position)
        if choose_label_i.shape == (3,):
            choose_label_i = np.reshape(choose_label_i,(1,3))
        choose_label.append(choose_label_i) 
    array_l = []
    for inpt in range(len(array)):
        B = np.zeros([256, 256],dtype=float)
        for label in choose_label:
            z = np.array([[choose_spec_r0[inpt][1],choose_spec_r0[inpt][2],choose_spec_r0[inpt][3]]])
            if (label == z).all(1).any() == True:
                rows = dict(zip(list(range(0, len(B))), list(array[inpt][1][0])))
                cols = dict(zip(list(range(0, len(B)+1)), list(array[inpt][3][0])))
                for row in range(256):
                    for i in range(len(label)):                   
                        if rows[row] == label[i,0]:
                            for col in range(256):
                                if cols[col]<label[i,1]<=cols[col+1]:
                                    B[col][row] = label[i,2]
        array_l.append(B)       
    choose_label2 = []
    for iu in range(len(array_l)):
        choose_label = array_l[iu]
        choose_label = choose_label[126:131,:] 
        for i in range(256):
            if choose_label[2,i] == 0.0 and choose_label[0,i] != 0.0:
                choose_label[2,i] = choose_label[0,i]
            if choose_label[2,i] == 0.0 and choose_label[1,i] != 0.0:
                choose_label[2,i] = choose_label[1,i]
            if choose_label[2,i] == 0.0 and choose_label[3,i] != 0.0:
                choose_label[2,i] = choose_label[3,i]     
            if choose_label[2,i] == 0.0 and choose_label[4,i] != 0.0:
                choose_label[2,i] = choose_label[4,i] 
        choose_label2.append(choose_label)
    return choose_label2

def iou(array, pred_array_int, choose_label):
    t_iou1 = []
    for index in range(len(choose_label)):
        N = array[index][1][0]
        for x in range(len(N[:])):
            if N[x] != 0:
                break
        for z in range(len(N[:])):
            if N[z] != 0:
                v = z
        for p1 in range(len(pred_array_int[index][2,x:v+1])):
            if pred_array_int[index][2,x:v+1][p1] != 0:
                break
        for p3 in range(len(pred_array_int[index][2,x:v+1])):
            if pred_array_int[index][2,x:v+1][p3] != 0:
                p2 = p3
        for t1 in range(len(choose_label[index][2,x:v+1])):
            if choose_label[index][2,x:v+1][t1] != 0:
                break
        for t3 in range(len(choose_label[index][2,x:v+1])):
            if choose_label[index][2,x:v+1][t3] != 0:
                t2 = t3
        rt_pi = N[x:v+1][p1:p2+1]
        rt_ti = N[x:v+1][t1:t2+1]
        int_pi = pred_array_int[index][2,x:v+1][p1:p2+1]
        int_ti = choose_label[index][2,x:v+1][t1:t2+1]
        rt_p = dict(zip(list(rt_pi), list(int_pi)))
        rt_t = dict(zip(list(rt_ti), list(int_ti)))
        inter_rt_int = list(rt_p.items() & rt_t.items())
        union_rt_int = list(rt_p.items() | rt_t.items())
        def take_int(elem):
            return elem[0]
        inter_rt_int.sort(key=take_int)
        union_rt_int.sort(key=take_int)
        inter_rti = np.array([i[0] for i in inter_rt_int])
        inter_inti = np.array([i[1] for i in inter_rt_int])
        union_rti = np.array([i[0] for i in union_rt_int])
        union_inti = np.array([i[1] for i in union_rt_int])
        if inter_rti.shape == (0,) or inter_inti.shape == (0,) or union_rti.shape == (0,) or union_inti.shape == (0,):
            iou = 0
        elif rt_pi[0] >= rt_ti[0] and rt_pi[-1] <= rt_ti[-1] and inter_rti.shape != (0,):
            inter = np.trapz(int_pi,rt_pi)
            union = np.trapz(int_ti,rt_ti)
            iou = inter/union
        elif (rt_pi[0] < rt_ti[0] and rt_pi[-1] > rt_ti[-1]) or (rt_pi[0] == rt_ti[0] and rt_pi[-1] > rt_ti[-1]) or rt_pi[0] < rt_ti[0] and rt_pi[-1] == rt_ti[-1]:
            inter = np.trapz(int_ti,rt_ti)
            union = np.trapz(int_pi,rt_pi)
            iou = inter/union
        elif rt_pi[0] > rt_ti[0] and rt_pi[-1] > rt_ti[-1] and inter_rti.shape != (0,): 
            inter = np.trapz(inter_inti,inter_rti)
            union = np.trapz(union_inti,union_rti)
            iou = inter/union
        elif rt_pi[0] < rt_ti[0] and rt_pi[-1] < rt_ti[-1] and inter_rti.shape != (0,): 
            inter = np.trapz(inter_inti,inter_rti)
            union = np.trapz(union_inti,union_rti)
            iou = inter/union
        t_iou1.append(round(iou,4))
    return t_iou1

if __name__ == '__main__':
    path = "./Metabolomics dataset/10.mzxml"
    choose_spec0, rt, rt_mean_interval = readms(path)
    choose_spec = get_range(choose_spec0,rt,rt_mean_interval, mass_inv = 1, rt_inv = 15, min_intensity=6000)
    p = Pool(5)
    array = p.map(get_array,choose_spec)
    
    model = DeepPIC_load('./best_unet2_zz.h5')
    
    preds = DeepPIC_predict(model, scaler(array),batch_size=4)
    pred_array_int = pred_array(0, preds, array)
    pred_array_mz = pred_array(2, preds, array)
    pics = pics(array, pred_array_int, pred_array_mz, choose_spec)
    
    plt.figure(figsize=(10,8))
    plt.plot(pics[0][:, 0], pics[0][:, 1], marker='o', color='b', linewidth=2.0, linestyle='--', label='linear line')
    plt.legend(loc='upper right')
    plt.xlabel('RT')
    plt.ylabel('intensity')
    plt.title("PIC")
    plt.show()
    
    
      
