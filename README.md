# DeepPIC
This is the code repo for the paper *Deep Learning-based Pure Ion Chromatogram Extraction for LC-MS*. We developed a deep learning-based pure ion chromatogram method (DeepPIC) for extracting PICs from raw data files directly and automatically. The DeepPIC method has already been integrated into the KPIC2 framework. The combination can provide the entire pipeline from raw data to discriminant models for metabolomic datasets.

<div align="center">
<img src="https://github.com/yuxuanliao/DeepPIC/blob/main/Workflow of the DeepPIC method.png" width=917 height=788 />
</div>

## Installation
**1.** Install [Anaconda](https://www.anaconda.com) for python 3.8.13.

**2.** Install [R 4.2.1](https://mirrors.tuna.tsinghua.edu.cn/CRAN).

**3.** Install KPIC2 in R language.

The method of installing KPIC2 can refer to https://github.com/hcji/KPIC2.
- First install the depends of KPIC2.
	```shell
  install.packages(c("BiocManager", "devtools", "Ckmeans.1d.dp", "Rcpp", "RcppArmadillo", "mzR", "parallel", "shiny", "plotly", "data.table", "GA", "IRanges",  "dbscan", "randomForest"))
  BiocManager::install(c("mzR","ropls"))
	```
- Then, download the source package of KPIC2 at [url](https://github.com/hcji/KPIC2/releases) and install the package locally.

**4.** Create environment and install main packages.

- Open commond line, create environment.
	```shell
  conda create --name DeepPIC python=3.8.13
  conda activate DeepPIC
	```
- Clone the repository and enter.
	```shell
  git clone https://github.com/yuxuanliao/DeepPIC.git
  cd DeepPIC
	```
- Install main packages in [requirements.txt](https://github.com/yuxuanliao/DeepPIC/blob/main/requirements.txt) with following commands.
  ```shell
  python -m pip install -r requirements.txt
	```
- Set environment variables for calling R language using rpy2.

  R_HOME represents the installation location of the R language.
  
  R_USER represents the installation location of the rpy2 package.
  ```shell
  setx "R_HOME" "C:\Program Files\R\R-4.2.1"
  setx "R_USER" "C:\Users\yxliao\anaconda3\Lib\site-packages\rpy2"
	```

## DeepPIC
The following files are in the [DeepPIC](DeepPIC) folder:
- [train.py](DeepPIC/train.py). for model training
- [extract.py](DeepPIC/extract.py). extract PICs from raw LC-MS files
- [predict.py](DeepPIC/predict.py). define the IoU metric for PICs and evalute the DeepPIC model

## KPIC2
The following files are in the [KPIC2](KPIC2) folder:
- [KPIC2.py](KPIC2/KPIC2.py). for integrating DeepPIC into KPIC2 to implement the whole process of metabolomics processing
- [KPIC2.R](KPIC2/KPIC2.R). the code for the feature detection, alignment, grouping, missing value filling, and building classification models
- [permutation_vip.py](KPIC2/permutation_vip.py). define some functions for file format conversion, permutation test, and biomarkers selection
- *[files](KPIC2/files)*:
    - *[pics](KPIC2/files/pics)* (PICs extracted by DeepPIC from each LC-MS file in the metabolomics dataset by running [extract.py](DeepPIC/extract.py))
    - *[scantime](KPIC2/files/scantime)* (RTs read from each LC-MS file in the metabolomics dataset using [OpenMS](https://github.com/OpenMS/OpenMS/releases))
    - *[KPIC2_result.csv](KPIC2/files/KPIC2_result.csv)* (the file generated by running [KPIC2.py](KPIC2/KPIC2.py))
    - *[KPIC2_result_plot.csv](KPIC2/files/KPIC2_result_plot.csv)* (the file format for the OPLS-DA scores plot, permutation test, and biomarkers selection by running Datatransform function in [permutation_vip.py](KPIC2/permutation_vip.py))
    
## Others
The following files are in the [others](others) folder:
- [metabolomics.py](others/metabolomics.py). the code for the OPLS-DA scores plot, permutation test, biomarkers selection and hierarchical cluster analysis
- [quantitative.py](others/quantitative.py). evaluate the quantitative ability of feature extraction methods
- [XCMS.R](others/XCMS.R). the code for XCMS to detect peaks
- *[Simulation](others/Simulation)*:
    - [mssimulator.py](others/Simulation/mssimulator.py). define some functions for generating the simulated LC-MS files
    - [simulated_mm48.py](others/Simulation/simulated_mm48.py). generate the simulated MM48 dataset
    
## Dataset
The dataset with 200 input-label pairs used to train, validate, and test the DeepPIC model is in the [dataset](dataset) folder. As the model and the data exceeded the limits, we have uploaded the optimized model and the datasets (MM48, simulated MM48, quantitative, metabolomics and different instrumental datasets) to [Github release page](https://github.com/yuxuanliao/DeepPIC/releases).

## Usage
The example code for model training is included in the [train.ipynb](train.ipynb).

The example code for feature extraction is included in the [extract.ipynb](extract.ipynb).

The example code for integrating DeepPIC into KPIC2 to implement the whole process of metabolomics processing is included in the [Integration_into_KPIC2.ipynb](Integration_into_KPIC2.ipynb).

## Start from raw LC-MS dataset to discriminant model
By running [extract.py](DeepPIC/extract.py), user can use DeepPIC to extract PICs from each LC-MS file in the metabolomics dataset. The whole process of metabolomics processing can be implemented by running [KPIC2.py](KPIC2/KPIC2.py) directly. Please refer to [extract.ipynb](extract.ipynb) and [Integration_into_KPIC2.ipynb](Integration_into_KPIC2.ipynb) for details. Thus, you can use DeepPIC+KPIC2 to process your data.

## Information of maintainers
- 212311021@csu.edu.cn
