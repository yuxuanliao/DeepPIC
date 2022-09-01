library("xcms")
mzML = "C:/Users/yxliao/Desktop/Smartgit/Zenodo_DeepPIC/Quantitative dataset/2012_02_03_PStd_000_2.mzML"
raw_data <- readMSData(mzML, mode = "onDisk", centroided = FALSE)
test_rt = head(rtime(raw_data),4001)
cwp <- CentWaveParam(peakwidth = c(5, 12), snthresh = 3, prefilter = c(1, 30))
xdata <- findChromPeaks(raw_data, param = cwp)
xdata
a = chromatogram(xdata)
b = chromPeaks(a)
b
write.table (b, file ="D:/Dpic/data/lyx/xcms_dl/0_xcms.txt")
head(chromPeaks(xdata),10000)
plotChromPeaks(xdata)