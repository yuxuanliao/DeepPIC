# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 21:12:13 2022

@author: yxliao
"""

import subprocess, sys, os
import pandas as pd
import numpy as np
sys.path.append(r'D:\Dpic\python')
from mssimulator import simulation, simulation2, data2mzxml,parse_featureXML_GT
from matplotlib import pyplot as plt
from FPIC import tic, toc
from FPIC import pics2peaks, merge_peaks
import pyopenms


def FeatureFindingMetabo(mzfile, noise_threshold_int, snr):
    finder = 'C:/Program Files/OpenMS-2.3.0/bin/FeatureFinderMetabo.exe'
    feature_file = 'tmp.featureXML'
    noise_threshold_int = noise_threshold_int / snr
    subprocess.call([finder, '-in', mzfile, '-out', feature_file, 
               '-algorithm:common:noise_threshold_int', f'{noise_threshold_int}',
               '-algorithm:common:chrom_peak_snr', f'{snr}',
               '-algorithm:common:chrom_fwhm', '10',
               '-algorithm:mtd:mass_error_ppm', '20',
               '-algorithm:mtd:reestimate_mt_sd', 'true',
               '-algorithm:mtd:min_sample_rate', '0',
               '-algorithm:mtd:min_trace_length', '2', ##1
               '-algorithm:epd:width_filtering', 'off',
               '-algorithm:ffm:charge_lower_bound', '1',
               '-algorithm:ffm:charge_lower_bound', '5'])  
    featuremap = pyopenms.FeatureMap()
    featurexml = pyopenms.FeatureXMLFile()
    featurexml.load(feature_file, featuremap)
    os.remove(feature_file)
    return featuremap

def parse_featureXML_FFM(featuremap):   
    df = pd.DataFrame(columns=['rt', 'mz', 'intensity'])   
    for i in range(featuremap.size()):
        feature = featuremap[i]
        isotope_distances = feature.getMetaValue(b'isotope_distances')
        rt = feature.getRT()
        mz = feature.getMZ()
        intensity = feature.getIntensity()
        for j in range(feature.getMetaValue(b'num_of_masstraces')):
            if j == 0:
                df.loc[len(df)] = [rt, mz, intensity]
            else:
                mz_delta = isotope_distances[j-1]
                mz = mz + mz_delta
                df.loc[len(df)] = [rt, mz, intensity] 
    return df

def pics2df(pics):
    df = pd.DataFrame(columns=['rt', 'mz', 'intensity'])
    for i,pic in enumerate(pics):
        idx = pic[:,2].argmax()
        rt  = pic[idx,0]
        mz  = pic[idx,1]
        intensity = pic[idx,2]
        df.loc[len(df)] = [rt, mz, intensity] 
    return df

def peaks2df(peaks):
    df = pd.DataFrame(columns=['rt', 'mz', 'intensity'])
    for i in range(peaks.shape[0]):
        rt  = peaks[i,3]
        mz  =peaks[i,0]
        intensity = peaks[i,6]
        df.loc[len(df)] = [rt, mz, intensity] 
    return df

def match_features(ground_truths, df):
    for i in range(len(df)):
        rt  = df.at[i, 'rt']
        mz  = df.at[i, 'mz']
        for j in range(len(ground_truths)):
            if(rt >= ground_truths.at[j, 'rt_min'] and rt <= ground_truths.at[j, 'rt_max'] and
               mz >= ground_truths.at[j, 'mz_min']-0.01 and mz <= ground_truths.at[j, 'mz_max']+0.01
               ): 
                ground_truths.at[j, 'detected'] = True
                ground_truths.at[j, 'pic_id'] = i

def metrics(TP, FN, FP):
    r = TP/(TP+FN)
    p = TP/(TP+FP)
    f1 = (2*r*p)/(r+p)
    return r, p, f1

# mm48_all = pd.read_csv('D:/Dpic/data/MM48_annotations.csv')
# mm48_all['charge'] = [1] * mm48_all.shape[0]
# mm48_all['shape'] = ['gauss'] * mm48_all.shape[0]
# mm48_all['source'] = ['ESI'] * mm48_all.shape[0]
# mm48 = mm48_all[['Name', 'Formel','RT','RT2','Intensity','charge','shape','source']]
# mm48.to_csv('D:/Dpic/data/MM48_MSSimulator.csv', header=False, index=False)

names = ['stddev','FFM_Recall', 'FFM_Precision', 'FFM_FScore']
results = pd.DataFrame(columns=names) 
parameters = [[0, 1.35],[1,13.5],[3, 40.5],[8, 108], [13, 175.5], [18, 243], [23, 310.5], [30, 405]]

openms_path = "C:/Program Files/OpenMS-2.3.0/bin/"

os.getcwd()
os.chdir('D:/Dpic/data')

for i,p in enumerate(parameters):   
    simulation('D:/Dpic/data/test.fasta','D:/Dpic/data/MM48_MSSimulator.csv', 'D:/Dpic/data/MM48_MSS_Profile0.mzML', 'D:/Dpic/data/MM48_MSS0.featureXML', 0) 
    peak_picker = 'C:/Program Files/OpenMS-2.3.0/bin/PeakPickerHiRes.exe'
    subprocess.call([peak_picker,'-in', 'D:/Dpic/data/MM48_MSS_Profile0.mzML','-out', 
                     'D:/Dpic/data/MM48_MSS0.mzML'])
    
    #
    data2mzxml('D:/Dpic/data/stddev/MM48_MSS_0.mzML')
    #
    ground_truths = parse_featureXML_GT('D:/Dpic/data/stddev/profile_feature/MM48_MSS30.featureXML')
        
        
        
    mzfile =  'D:/Dpic/data/stddev/MM48_MSS_30.mzxml'
    mzMLfile =  'D:/Dpic/data/stddev/MM48_MSS_30.mzML'
    
    tic()
    feature_map = FeatureFindingMetabo(mzMLfile, 405, 3)
    df_ffm = parse_featureXML_FFM(feature_map)
    toc()
    
    match_ffm = ground_truths.copy()
    match_features(match_ffm, df_ffm)
    m = match_ffm.detected.value_counts().values
    
    
    df_xcms = pd.read_csv('D:/Dpic/data/stddev/5-12/30_df_xcms.csv')
    match_xcms = ground_truths.copy()
    match_features(match_xcms, df_xcms)
    x = match_xcms.detected.value_counts().values
    
    # from r_functions import XCMS
    # tic()
    # df_xcms = XCMS(mzMLfile, w1=5, w2=50, snr=3, intensity=405)
    # toc()
    # match_xcms = ground_truths.copy()
    # match_features(match_xcms, df_xcms)
    # x = match_xcms.detected.value_counts().values
    
        
    m_r, m_p,m_f = metrics(m[0], m[1], df_ffm.shape[0]-m[0])
    x_r, x_p,x_f = metrics(x[0], x[1], df_xcms.shape[0]-x[0])
        
        
    results.loc[len(results)] = [8,m_r,m_p,m_f] 
    result_rd = results.round(4) 