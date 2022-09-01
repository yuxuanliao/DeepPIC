# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 21:06:48 2022

@author: yxliao
"""

import subprocess, sys, os
import pyopenms
import pandas as pd

def simulation(fasta, contaminants, out, out_cntm, stddev,
               simulator = 'C:/Program Files/OpenMS-2.3.0/bin/MSSimulator.exe'):   
    """
        Should copy "C:\Program Files\OpenMS\share\OpenMS\examples" to working directory of Python
    """ 
   
    subprocess.call([simulator, '-in', fasta, '-out', out, '-out_cntm',out_cntm, 
               '-algorithm:MSSim:RawSignal:contaminants:file', contaminants,
               '-algorithm:MSSim:RawSignal:noise:detector:stddev', f'{stddev}',
               '-algorithm:MSSim:RawSignal:resolution:value', '5000',
               '-algorithm:MSSim:RawSignal:resolution:type', 'constant',
               '-algorithm:MSSim:Ionization:mz:lower_measurement_limit', '10',
               '-algorithm:MSSim:Ionization:mz:upper_measurement_limit', '1000',
               '-algorithm:MSSim:RT:total_gradient_time', '1000',
               '-algorithm:MSSim:RT:sampling_rate', '0.25',
               '-algorithm:MSSim:RT:scan_window:min', '0',
               '-algorithm:MSSim:RT:scan_window:max', '1000'])

def simulation2(fasta, contaminants, out, out_cntm,
               simulator = 'C:/Program Files/OpenMS-2.3.0/bin/MSSimulator.exe'):   
    """
        Should copy "C:\Program Files\OpenMS\share\OpenMS\examples" to working directory of Python
    """ 

    subprocess.call([simulator, '-in', fasta, '-out', out, '-out_cntm',out_cntm, 
               '-algorithm:MSSim:RawSignal:contaminants:file', contaminants,
               '-algorithm:MSSim:RawSignal:resolution:value', '5000',
               '-algorithm:MSSim:RawSignal:resolution:type', 'constant',
               '-algorithm:MSSim:Ionization:mz:lower_measurement_limit', '10',
               '-algorithm:MSSim:Ionization:mz:upper_measurement_limit', '1000',
               '-algorithm:MSSim:RT:total_gradient_time', '1000',
               '-algorithm:MSSim:RT:sampling_rate', '0.25',
               '-algorithm:MSSim:RT:scan_window:min', '0',
               '-algorithm:MSSim:RT:scan_window:max', '1000'])
    
def data2mzxml(path, converter = 'C:/Program Files/OpenMS-2.3.0/bin/FileConverter.exe'):
    if os.path.isfile(path):
        files = [path]
        path = ""
    elif os.path.isdir(path):
        files=os.listdir(path)
    for f in files:
        if f.lower().endswith(".mzdata"): 
            file_in  = path + f
            file_out = path + f[0:-6] + "mzxml"
            subprocess.call([converter, '-in', file_in, '-out', file_out])
        if f.lower().endswith(".mzml"):
            file_in  = path + f
            file_out = path + f[0:-4] + "mzxml"
            subprocess.call([converter, '-in', file_in, '-out', file_out])
            
def parse_featureXML_GT(feature_file):
    featuremap = pyopenms.FeatureMap()
    featurexml = pyopenms.FeatureXMLFile()
    featurexml.load(feature_file, featuremap)
    
    hulls = pd.DataFrame(columns=['rt_min', 'rt_max', 'mz_min', 'mz_max', 'detected', 'pic_id'])   
    for i in range(featuremap.size()):
        feature = featuremap[i]
        chs = feature.getConvexHulls()
        for j in range(len(chs)):
            pts = chs[j].getHullPoints()
            hulls.loc[len(hulls)] = [pts.min(0)[0], pts.max(0)[0], pts.min(0)[1], pts.max(0)[1], False, -1]
    return hulls

