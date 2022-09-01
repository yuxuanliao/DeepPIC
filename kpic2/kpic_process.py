# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 09:19:22 2022

@author: yxliao
"""

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from pyopls import OPLS
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import r2_score, accuracy_score
from permutation_vip import *
#permutation
import warnings
from sys import stderr
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import is_classifier, clone, ClassifierMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import check_cv, cross_val_predict
from sklearn.utils import indexable, check_random_state
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
plt.rc('font',family='Calibri') 


if __name__ == '__main__':
    
    pics5_ffn = "D:/Dpic/data2/leaf_seed/pics/pics01"
    pics5_dir = "D:/Dpic/data2/leaf_seed/pics/pics01/"
    pics5_path1 = "D:/Dpic/data2/leaf_seed/data/1.mzXML"
    pics5_ps = "D:/Dpic/data2/leaf_seed/scantime/scantime01/rt1.txt"
    
    pics4_ffn = "D:/Dpic/data2/leaf_seed/pics/pics02"
    pics4_dir = "D:/Dpic/data2/leaf_seed/pics/pics02/"
    pics4_path2 = "D:/Dpic/data2/leaf_seed/data/2.mzXML"
    pics4_ps = "D:/Dpic/data2/leaf_seed/scantime/scantime02/rt2.txt"
    
    pics_1_ff = "D:/Dpic/data2/leaf_seed/pics"
    pics_1_fps = "D:/Dpic/data2/leaf_seed/scantime"
    pics_1_fp = "D:/Dpic/data2/leaf_seed/data/"
    

    numpy2ri.activate()
    robjects.r('''source('kpic_process.R')''')
    kpic_pics5 = robjects.globalenv['kpic_pics5']
    PICset_decpeaks = robjects.globalenv['PICset_decpeaks']
    PICset_split = robjects.globalenv['PICset_split']
    PICset_getPeaks = robjects.globalenv['PICset_getPeaks']
    PICset_group = robjects.globalenv['PICset_group']
    PICset_align1 = robjects.globalenv['PICset_align1']
    PICset_align2 = robjects.globalenv['PICset_align2']
    PICset_align3 = robjects.globalenv['PICset_align3']
    kpic_iso = robjects.globalenv['kpic_iso']
    kpic_mat = robjects.globalenv['kpic_mat']
    kpic_fill = robjects.globalenv['kpic_fill']
    kpic_pattern = robjects.globalenv['kpic_pattern']
    #kpic_pattern = robjects.globalenv['kpic_pattern']
    pics_1 = kpic_pics5(pics5_ffn, pics5_dir, pics5_path1, pics5_ps, pics4_ffn, pics4_dir, pics4_path2, pics4_ps, pics_1_ff, pics_1_fps, pics_1_fp)
    PICS = PICset_decpeaks(pics_1)
    PICS = PICset_split(PICS)
    PICS = PICset_getPeaks(PICS)
    groups_raw = PICset_group(PICS)
    groups_align1 = PICset_align1(groups_raw)
    groups_align2 = PICset_align2(groups_align1)
    groups_align3 = PICset_align2(groups_align2)
    groups_align = kpic_iso(groups_align3)
    data = kpic_mat(groups_align)
    data = kpic_fill(data)
    result = kpic_pattern(data, file1 = "C:/Users/yxliao/Desktop/Smartgit3/DeepPIC/s111.csv")
    #data = kpic_datatf(data)
    #df = pd.DataFrame(data)
    #df.to_csv("D:/Dpic/data2/leaf_seed/s111_lyx20220831.csv", index=True, header=True)
    
    #OPLS-DA
    data = pd.read_csv("C:/Users/yxliao/Desktop/Smartgit3/DeepPIC/files/s_python.csv",encoding='gbk')
    X = np.array(data.values[:,1:].T,dtype=float)
    Y= np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])  #20个样品对应的标签
    opls = OPLS(1)
    Z = opls.fit_transform(X, Y)
    pls = PLSRegression(1)
    uncorrected_r2 = r2_score(Y, pls.fit(X, Y).predict(X))
    corrected_r2 = r2_score(Y, pls.fit(Z, Y).predict(Z))
    uncorrected_q2 = r2_score(Y, cross_val_predict(pls, X, Y, cv=LeaveOneOut()))
    corrected_q2 = r2_score(Y, cross_val_predict(pls, Z, Y, cv=LeaveOneOut()))
    pls.fit_transform(Z, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bwith = 2
    TK = plt.gca()
    TK.spines['bottom'].set_linewidth(bwith)
    TK.spines['left'].set_linewidth(bwith)
    TK.spines['top'].set_linewidth(bwith)
    TK.spines['right'].set_linewidth(bwith)

    font = {'family' : 'Calibri',
    'weight' : 'normal',
    'size'   : 40,
        }

    df = pd.DataFrame(np.column_stack([pls.x_scores_, opls.T_ortho_[:, 0]]),
                      index=Y,columns=['t', 't_ortho'])                           
    pos_df = df[Y==0]
    neg_df = df[Y==1]
    plt.scatter(neg_df['t'], neg_df['t_ortho'], s=90, c='g', marker='o', label='seed')
    plt.scatter(pos_df['t'], pos_df['t_ortho'], s=90, c='red',marker='^', label='leaf')

    ax.set_title('Scores (OPLS-DA)',fontproperties = 'Calibri', size = 45)
    ax.set_xlabel('t',font)
    ax.set_ylabel("t_ortho",font)    

    plt.yticks(fontproperties = 'Calibri', size = 30)
    plt.xticks(fontproperties = 'Calibri', size = 30)

    plt.legend(loc = 'best', prop={'family' : 'Calibri', 'size' : 25})
    
    #permutation
    kf = KFold(n_splits = 5, shuffle=True, random_state=420) 
    permutation_scores = permutation_test_score(pls, Z, Y, groups= None , cv=kf,
                               n_permutations=2000, n_jobs=1, random_state=420,
                               verbose=0, fit_params=None)
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    bwith = 2 #边框宽度设置为2
    TK = plt.gca()#获取边框
    TK.spines['bottom'].set_linewidth(bwith)#图框下边
    TK.spines['left'].set_linewidth(bwith)#图框左边
    TK.spines['top'].set_linewidth(bwith)#图框上边
    TK.spines['right'].set_linewidth(bwith)#图框右边

    font = {'family' : 'Calibri',
    'weight' : 'normal',
    'size'   : 40,
        }

    ax2.hist(permutation_scores[0][1], bins=50, density=False, fc="lightpink", ec="magenta")
    ax2.axvline(permutation_scores[0][0], linewidth = 3, ls="--", color="deepskyblue")

    ax2.set_xlabel('$\mathregular{Q^2}$',font)
    ax2.set_ylabel("Frequency",font)    

    plt.yticks(fontproperties = 'Calibri', size = 30)
    plt.xticks(fontproperties = 'Calibri', size = 30)
    
    #vip
    import tensorflow as tf
    import numpy as np
    import csv
    import os
    import bisect
    import pandas as pd
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
     
    DATA = pd.read_csv("C:/Users/yxliao/Desktop/Smartgit3/DeepPIC/files/s_python.csv", encoding='gbk')

    X = np.array(DATA.values[:,1:].T,dtype=float)
    COMs = DATA.values[:,0]

    VIPs = vip(Z, Y, pls)   

    COM = []
    VIP = []

    for vv in range(11):
        if VIPs[vv]>=1:
            
            VIP.append(VIPs[vv])
            COM.append(COMs[vv])
     

    sorted_vips = sorted(enumerate(VIP), key=lambda x: x[1])
    idx = [i[0] for i in sorted_vips]
    vips = [i[1] for i in sorted_vips]
    COMS=[]
    for jj in range(len(COM)):
        COMS.append(COM[idx[jj]])

    fig = plt.figure()
    ax11 = fig.add_subplot(111)
    bwith = 2
    TK = plt.gca()
    TK.spines['bottom'].set_linewidth(bwith)
    TK.spines['left'].set_linewidth(bwith)
    TK.spines['top'].set_linewidth(bwith)
    TK.spines['right'].set_linewidth(bwith)

    font = {'family' : 'Calibri',
    'weight' : 'normal',
    'size'   : 40,
        }
    plt.grid(axis="y",linestyle='-.')
    plt.scatter(vips,range(len(COMS)),c="#88c999",s=250,marker="o")
    ax11.set_xlabel('VIP',font)
    plt.yticks(range(len(COMS)),COMS,fontproperties = 'Calibri', size = 30)
    plt.xticks(fontproperties = 'Calibri', size = 30)
    
    
    

