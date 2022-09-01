# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:44:59 2022

@author: yxliao
"""

#permutation
import warnings
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import is_classifier, clone, ClassifierMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import check_cv, cross_val_predict
from sklearn.utils import indexable, check_random_state


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