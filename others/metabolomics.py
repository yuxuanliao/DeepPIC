# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 10:45:05 2022

@author: yxliao
"""

from pyopls import OPLS
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
plt.rc('font',family='Calibri') 

#OPLS-DA
data = pd.read_csv("../KPIC2/files/KPIC2_result_plot.csv",encoding='gbk')   #
X = np.array(data.values[:,1:].T,dtype=float)
Y= np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])  #20labels
opls = OPLS(1)
Z = opls.fit_transform(X, Y)
pls = PLSRegression(1)
uncorrected_r2 = r2_score(Y, pls.fit(X, Y).predict(X))
corrected_r2 = r2_score(Y, pls.fit(Z, Y).predict(Z))
uncorrected_q2 = r2_score(Y, cross_val_predict(pls, X, Y, cv=LeaveOneOut()))
corrected_q2 = r2_score(Y, cross_val_predict(pls, Z, Y, cv=LeaveOneOut()))
pls.fit_transform(Z, Y)

fig = plt.figure()
ax = fig.add_subplot(231)
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

def _permutation_test_score(estimator, X, y, groups=None, cv='warn',
                            n_jobs=None, verbose=0, fit_params=None,
                            pre_dispatch='2*n_jobs', method='predict',
                            score_functions=None):
    """Auxiliary function for permutation_test_score"""
    if score_functions is None:
        score_functions = [r2_score]
    y_pred = cross_val_predict(estimator, X, y, groups, cv, n_jobs, verbose, 
                               fit_params, pre_dispatch, method)
    cv_scores = [score_function(y, y_pred) for score_function in score_functions]
    return np.array(cv_scores)

def _shuffle(y, groups, random_state):
    """Return a shuffled copy of y eventually shuffle among same groups."""
    if groups is None:
        indices = random_state.permutation(len(y))
    else:
        indices = np.arange(len(groups))
        for group in np.unique(groups):
            this_mask = (groups == group)
            indices[this_mask] = random_state.permutation(indices[this_mask])
    return safe_indexing(y, indices)

def safe_indexing(X, indices):
    if hasattr(X, "iloc"):
        indices = indices if indices.flags.writeable else indices.copy()
        try:
            return X.iloc[indices]
        except ValueError:
            warnings.warn("Copying input dataframe for slicing.",
                          DataConversionWarning)
            return X.copy().iloc[indices]
    elif hasattr(X, "shape"):
        if hasattr(X, 'take') and (hasattr(indices, 'dtype') and
                                   indices.dtype.kind == 'i'):
            # This is often substantially faster than X[indices]
            return X.take(indices, axis=0)
        else:
            return X[indices]
    else:
        return [X[idx] for idx in indices]
    
def permutation_test_score(estimator, X, y, groups=None, cv='warn',
                           n_permutations=100, n_jobs=None, random_state=0,
                           verbose=0, pre_dispatch='2*n_jobs', cv_score_functions=None,
                           fit_params=None, method='predict', parallel_by='permutation'):
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    random_state = check_random_state(random_state)
    if cv_score_functions is None:
        if isinstance(estimator, ClassifierMixin):
            cv_score_functions = [accuracy_score]
        else:
            cv_score_functions = [r2_score]
            
    score = _permutation_test_score(clone(estimator), X, y, groups, cv,
                                    n_jobs, verbose, fit_params, pre_dispatch,
                                    method, cv_score_functions)
    if parallel_by == 'estimation':
        permutation_scores = np.vstack([
            _permutation_test_score(
                clone(estimator), X, _shuffle(y, groups, random_state),
                groups, cv, n_jobs, verbose, fit_params, pre_dispatch,
                method, cv_score_functions
            ) for _ in range(n_permutations)
        ])
    elif parallel_by == 'permutation':
        permutation_scores = np.vstack(
            Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)(
                delayed(_permutation_test_score)(
                    clone(estimator), X, _shuffle(y, groups, random_state),
                    groups, cv, fit_params=fit_params, method=method, score_functions=cv_score_functions
                ) for _ in range(n_permutations)
            )
        )
    else:
        raise ValueError(f'Invalid option for parallel_by {parallel_by}')
    pvalue = (np.sum(permutation_scores >= score, axis=0) + 1.0) / (n_permutations + 1)
    return [(score[i], permutation_scores[:, i], pvalue[i]) for i in range(len(score))]
    # return score, permutation_scores, pvalue
    
from sklearn.model_selection import KFold
kf = KFold(n_splits = 5, shuffle=True, random_state=420) 
permutation_scores = permutation_test_score(pls, Z, Y, groups= None , cv=kf,
                           n_permutations=2000, n_jobs=1, random_state=420,
                           verbose=0, fit_params=None)

import matplotlib.pyplot as plt
ax2 = fig.add_subplot(232)
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
    
def vip(x, y, model):  
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_

    m, p = x.shape
    _, h = t.shape

    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)

    return vips


import matplotlib.pyplot as plt
import seaborn as sns  
 
DATA = pd.read_csv("../KPIC2/files/KPIC2_result_plot.csv", encoding='gbk')  #

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
ax11 = fig.add_subplot(231)
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

    
#HCA
import matplotlib.pyplot as plt     

DATA = pd.read_csv("C:/Users/yxliao/Desktop/Smartgit3/DeepPIC/hca_python.csv", encoding='gbk')  #

X = np.array(DATA.values[:,1:].T,dtype=float)
COM_vip = ['V2','V3','V4','V5','V6','V7','V8']        
sns.set(font_scale=2.2)
df = pd.DataFrame(X,index=None,columns = COM_vip)
cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
g = sns.clustermap(df.T,cmap="YlGnBu",col_cluster=False,row_cluster=True,annot_kws={"size": 30},standard_scale=1, linewidths = 0.5,  
                   cbar_kws=dict(orientation='horizontal'),figsize=(15,8),xticklabels=False)

x0, _y0, _w, _h = g.cbar_pos
g.ax_cbar.set_position([0.26, 0.88, 0.64, 0.02])
g.ax_cbar.tick_params(axis='x', length=10)
for spine in g.ax_cbar.spines:
    g.ax_cbar.spines[spine].set_color('crimson')
    g.ax_cbar.spines[spine].set_linewidth(2)

