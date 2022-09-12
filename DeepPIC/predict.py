# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:17:57 2022

@author: yxliao
"""

import os
import numpy as np
from DeepPIC.train import *
from DeepPIC.extract import *


def DeepPIC_predict(model, data, batch_size):
    preds = model.predict(data, batch_size, verbose=1)
    return preds

def DeepPIC_evaluate(data_input, data_label):
    ev = model.evaluate(data_input, data_label, verbose=1)
    return ev

def pred_array_test(x_test,preds):
    choose_spec2 = []
    preds_array = get_pred_array(preds)
    for iu in range(len(x_test)):
        choose_spec11 = x_test[iu]
        for i in range(256):
            for j in range(256):
                if preds_array[iu][i,j,0] == 0.0:
                    choose_spec11[i,j,0] = 0.0 
        choose_spec1 = choose_spec11[126:131,:] 
        choose_spec1 = np.squeeze(choose_spec1)
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
        for q3 in range(256):
            if preds_array[iu][128,q3] == 1:
                q2 = q3
        choose_spec1[2,0:q1] = 0
        choose_spec1[2,q2+1:] = 0
        choose_spec2.append(choose_spec1)
    return choose_spec2

def choose_label_test(y_test):
    choose_label2 = []
    for iu in range(len(y_test)):
        choose_label = y_test[iu]
        choose_label = choose_label[126:131,:] 
        choose_label=np.squeeze(choose_label)
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
        elif (rt_pi[0] < rt_ti[0] and rt_pi[-1] > rt_ti[-1]) or (rt_pi[0] == rt_ti[0] and rt_pi[-1] > rt_ti[-1]) or rt_pi[0] < rt_ti[0] and rt_pi[-1] == rt_ti[-1]:#注意等于号
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
    #test set
    path_pi = './dataset/positive samples/inputs'
    path_pl = './dataset/positive samples/labels'
    path_ni = './dataset/negative samples/inputs'
    array_p = PI(path_pi)[0]
    array_pl = PL(path_pl, path_pi)
    array_zs = NI(path_ni)
    array_zsl = NL()
    array_p.extend(array_zs)
    array_pl.extend(array_zsl)
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_data(array_p, array_pl)
    
    model = DeepPIC_load('./best_unet2_zz.h5')
    
    ev = DeepPIC_evaluate(x_test, y_test)
    preds_ts = DeepPIC_predict(model, x_test, batch_size=1)
    pred_array_int_ts = pred_array_test(x_test, preds_ts) #prediction results
    choose_label2_ts = choose_label_test(y_test) #labels
    
    #Mouse liver tissue
    path = './Different instrumental dataset/Mouse liver tissue/liver24_3.mzXML'
    choose_spec0, rt, rt_mean_interval = readms(path)
    choose_spec = get_range(choose_spec0,rt,rt_mean_interval,mass_inv = 0.08, rt_inv = 30, min_intensity=5000)
    import random
    random.seed (420)#Randomly take 200
    choose_spec_r = random.sample(choose_spec[0:358], 100)#100000
    choose_spec_r2 = random.sample(choose_spec[358:1970], 100)#5000
    choose_spec_r.extend(choose_spec_r2)
    def take_int(elem):
        return elem[3]
    choose_spec_r.sort(key=take_int)
    choose_spec_r4 = choose_spec_r[121:]#high200000
    choose_spec_r5 = choose_spec_r[58:121]#medium
    choose_spec_r6 = choose_spec_r[0:58]#low
    def take_int(elem):
        return elem[1]
    choose_spec_r4.sort(key=take_int)   #choose_spec_r4 5 6
    p = Pool(5)
    array = p.map(get_array,choose_spec_r4)  #choose_spec_r4 5 6
    
    preds = DeepPIC_predict(model, scaler(array), batch_size=1)
    pred_array_int = pred_array(0, preds, array)
    pred_array_mz = pred_array(2, preds, array)
    pics = pics(array, pred_array_int, pred_array_mz, choose_spec_r4) #r4 5 6 pics(array,pred_array_int,pred_array_mz,choose_spec)
    #label
    path = './Different instrumental dataset/Mouse liver tissue/high SNR features (-20)'
    choose_label2 = choose_label2(path, choose_spec_r4, array)
    t_iou1 = iou(array, pred_array_int, choose_label2)
    
    plt.figure(figsize=(10,8))
    plt.plot(pics[0][:, 0], pics[0][:, 1], marker='o', color='b', linewidth=2.0, linestyle='--', label='linear line')
    plt.legend(loc='upper right')
    plt.xlabel('RT')
    plt.ylabel('intensity')
    plt.title("PIC")
    plt.show()
    
    
    
        



    
    
    


