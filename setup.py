# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:20:06 2022

@author: yxliao
"""

from setuptools import setup, find_packages
from os import path
from io import open

this_directory = path.abspath(path.dirname(__file__))

def readme():
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        return f.read()
    
with open(path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="DeepPIC", 
    packages = ['DeepPIC'],
    package_data={'DeepPIC': ['ESPF/*']},
    version="1.0",
    author="Yuxuan Liao",
    license="BSD-3-Clause",
    author_email="212311021@csu.edu.cn",
    description="Deep Learning-based Pure Ion Chromatogram Extraction for LC-MS",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yuxuanliao/DeepPIC",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
