library(KPIC)

getNoise <- function(peaks, cwt2d, ridges){
  row_one <- row_one_del <- cwt2d[1,]
  del <- which(abs(row_one) < 10e-5)
  if (length(del)>0){
    row_one_del <- row_one[-del]
  }
  
  t <- 3*median(abs(row_one_del-median(row_one_del)))/0.67
  row_one[row_one > t] <- t
  row_one[row_one < -t] <- -t
  
  noises <- sapply(1:length(peaks),function(s){
    hf_win <- length(ridges$ridges_rows)
    win_s <- max(1, peaks[s] - hf_win)
    win_e <- min(ncol(cwt2d), peaks[s] + hf_win)
    return(as.numeric(quantile(abs(row_one[win_s:win_e]),0.9)))
  })
  return(noises)
}

cwtft <- function(val) {
  .Call('_KPIC_cwtft', PACKAGE = 'KPIC', val)
}

ridgesDetection <- function(cwt2d, val) {
  .Call('_KPIC_ridgesDetection', PACKAGE = 'KPIC', cwt2d, val)
}

peaksPosition <- function(val, ridges, cwt2d) {
  .Call('_KPIC_peaksPosition', PACKAGE = 'KPIC', val, ridges, cwt2d)
}

getSignal <- function(cwt2d, ridges, peaks) {
  .Call('_KPIC_getSignal', PACKAGE = 'KPIC', cwt2d, ridges, peaks)
}

peak_detection <- function(vec, min_snr, level=0){
  cwt2d <- cwtft(vec)
  sca <- cwt2d$scales
  cwt2d <- cwt2d$cwt2d
  ridges <- ridgesDetection(cwt2d, vec)
  if (length(ridges$ridges_rows)<1){return(NULL)}
  peaks <- peaksPosition(vec, ridges, cwt2d)
  signals <- getSignal(cwt2d, ridges, peaks)
  lens <- signals$ridge_lens
  lens[lens<0] <- 0
  scales <- sca[1+lens]
  lens <- signals$ridge_lens
  signals <- signals$signals
  peaks <- peaks+1
  noises <- getNoise(peaks, cwt2d, ridges)
  snr <- (signals+10^-5)/(noises+10^-5)
  refine <- snr>min_snr & lens>3 & vec[peaks]>level
  
  info <- cbind(peaks, scales, snr)
  info <- info[refine,]
  info <- unique(info)
  if (length(info)==0){return(NULL)
  } else if (length(info)>3){
    info <- info[order(info[,1]),]
    peakIndex=info[,1]; peakScale=info[,2]; snr=info[,3]; signals=vec[info[,1]]
  } else {
    peakIndex=info[1]; peakScale=info[2]; snr=info[3]; signals=vec[info[1]]
  }
  return(list(peakIndex=peakIndex, peakScale=peakScale, snr=snr, signals=signals))
}

decPeak <- function(picss, min_snr=6, level=0){
  peaks <- lapply(picss$pics,function(pic){
    peak_detection(pic[,2], min_snr, level)
  })
  
  nps <- sapply(peaks,function(peaki){
    length(peaki$peakIndex)
  })
  pics <- picss[["pics"]][nps>0]
  peaks <- peaks[nps>0]
  gc()
  
  picss[["pics"]] <- pics
  picss[["peaks"]] <- peaks
  output <- list(path=picss[["path"]], scantime=picss[["scantime"]], 
                 pics=picss$pics, peaks=picss$peaks)
  return(output)
}

PICset_decpeaks <- function(picset, min_snr=6, level=0){
  for (i in 1:length(picset)){
    picset[[i]] <- decPeak(picset[[i]])
  }
  return(picset)
}

integration <- function(x,yf){
  n <- length(x)
  integral <- 0.5*sum((x[2:n] - x[1:(n-1)]) * (yf[2:n] + yf[1:(n-1)]))
  return(integral)
}

getPeaks <- function(pics){
  mzinfo <- lapply(pics$pics,function(pic){
    mz <- mean(pic[,3], na.rm=TRUE)
    mzmin <- min(pic[,3], na.rm=TRUE)
    mzmax <- max(pic[,3], na.rm=TRUE)
    mzrsd <- sd(pic[,3], na.rm=TRUE)/mz*10^6
    c(mz,mzmin,mzmax,mzrsd)
  })
  
  rt <- sapply(pics$pics,function(pic){
    pic[which.max(pic[,2]),1]
  })
  
  snr <- sapply(pics$peaks,function(peaki){
    peaki$snr[which.max(peaki$signals)]
  })
  snr <- round(snr,2)
  
  maxo <- sapply(pics$pics,function(pic){
    max(pic[,2])
  })
  
  rtmin <- sapply(pics$pics,function(pic){
    pic[1,1]
  })
  rtmax <- sapply(pics$pics,function(pic){
    pic[nrow(pic),1]
  })
  
  area <- sapply(pics$pics,function(pic){
    round(integration(pic[,1],pic[,2]))
  })
  
  mzinfo <- round(do.call(rbind,mzinfo),4)
  colnames(mzinfo) <- c('mz','mzmin','mzmax','mzrsd')
  
  peakinfo <- cbind(rt,rtmin,rtmax,mzinfo,maxo,area,snr)
  pics$peakinfo <- peakinfo
  
  return(pics)
}

PICset_getPeaks <- function(picset){
  for (i in 1:length(picset)){
    picset[[i]] <- getPeaks(picset[[i]])
  }
  return(picset)
}

PICset_split <- function(PICS){
  PICS <- PICset.split(PICS)
  return(PICS)
}

PICset_group <- function(PICS, tolerance = c(0.01, 10)){
  groups_raw <- PICset.group(PICS, tolerance = tolerance)
  return(groups_raw)
}

PICset_align1 <- function(groups_raw, method = 'fftcc', move = 'loess'){
  groups_align1 <- PICset.align(groups_raw, method= method, move=move)
  return(groups_align1)
}

PICset_align2 <- function(groups_align1, tolerance = c(0.01, 10)){
  groups_align2 <- PICset.group(groups_align1$picset, tolerance = tolerance)
  return(groups_align2)
}

PICset_align3 <- function(groups_align2, method = 'fftcc', move = 'direct'){
  groups_align3 <- PICset.align(groups_align2, method= method, move=move)
  return(groups_align3)
}

kpic_iso <- function(groups_align3){
  groups_align4 <- groupCombine(groups_align3, type='isotope')
  return(groups_align4)
}

kpic_mat <- function(groups_align4){
  data <- getDataMatrix(groups_align4)
  return(data)
}

kpic_fill <- function(data){
  data <- fillPeaks.EIBPC(data)
  return(data)
}

kpic_datatf <- function(data){
  datatf <- data$data.mat
  return(datatf)
}

kpic_pattern <- function(data, file1){
  labels <- c(rep('leaf',10), rep('seed',10))
  analyst.OPLS(labels, data$data.mat)
  write.csv(data$data.mat, file=file1,
            quote=T, row.names = T)
}

kpic_pics5 <- function(pics5_ffn = "C:/Users/yxliao/Desktop/Smartgit3/DeepPIC/files/pics/pics01", pics5_dir = "C:/Users/yxliao/Desktop/Smartgit3/DeepPIC/files/pics/pics01/", pics5_path1 = "D:/Dpic/data2/leaf_seed/data/1.mzXML", pics5_ps = "C:/Users/yxliao/Desktop/Smartgit3/DeepPIC/files/scantime/scantime01/rt1.txt", pics4_ffn = "C:/Users/yxliao/Desktop/Smartgit3/DeepPIC/files/pics/pics02", pics4_dir = "C:/Users/yxliao/Desktop/Smartgit3/DeepPIC/files/pics/pics02/", pics4_path2 = "D:/Dpic/data2/leaf_seed/data/2.mzXML", pics4_ps = "C:/Users/yxliao/Desktop/Smartgit3/DeepPIC/files/scantime/scantime02/rt2.txt" , pics_1_ff = "C:/Users/yxliao/Desktop/Smartgit3/DeepPIC/files/pics", pics_1_fps = "C:/Users/yxliao/Desktop/Smartgit3/DeepPIC/files/scantime", pics_1_fp="D:/Dpic/data2/leaf_seed/data/"){
  first_file_name <- list.files(pics5_ffn)      
  dir <- paste(pics5_dir,first_file_name,sep = "")            
  n <- length(dir) 
  data1 <- read.table(file = dir[1])
  data2 <- as.matrix(data1)
  data<-list(data2)
  for (i in 2:n){
    new_data1 = read.table(file = dir[i])
    new_data2 = as.matrix(new_data1)
    new_data = list(new_data2)
    data = c(data, new_data)
  }
  pics1 <- list(data)
  path1 <- pics5_path1
  path_scantime <- pics5_ps
  scantime11 <- read.table(path_scantime)
  scantime22 <- as.matrix(scantime11)
  scantime1 <- list(scantime22)
  pics5 = list(path = path1,scantime = scantime1[[1]],
               pics=pics1[[1]])
  first_file_name <- list.files(pics4_ffn)      
  dir <- paste(pics4_dir,first_file_name,sep = "")            
  n <- length(dir) 
  data1 <- read.table(file = dir[1])
  data2 <- as.matrix(data1)
  data<-list(data2)
  for (i in 2:n){
    new_data1 = read.table(file = dir[i])
    new_data2 = as.matrix(new_data1)
    new_data = list(new_data2)
    data = c(data, new_data)
  }
  pics2 <- list(data)
  #p<-list(path=path,pics=pics[[1]])
  path2 <- pics4_path2
  path_scantime <- pics4_ps
  scantime11 <- read.table(path_scantime)
  scantime22 <- as.matrix(scantime11)
  scantime2 <- list(scantime22)
  pics4 = list(path = path2,scantime = scantime2[[1]],
               pics=pics2[[1]])
  pics_1 = list(pics5, pics4)
  #循环
  filenames <- dir(pics_1_ff, full.names = T)
  path_scantime <- dir(pics_1_fps, full.names = T)
  for(j in 3:length(filenames)){
    path <- paste(pics_1_fp,j,".mzXML",sep = "")
    dir <-dir(filenames[j], full.names = T)
    rtdir <-dir(path_scantime[j], full.names = T)
    scantime11 <- read.table(rtdir[1])
    scantime22 <- as.matrix(scantime11)
    scantime <- list(scantime22)
    data1 <- read.table(file = dir[1])
    data2 <- as.matrix(data1)
    data<-list(data2)
    for(i in 2:length(dir)){
      n <- length(dir) 
      data1 <- read.table(file = dir[i])
      data2 <- as.matrix(data1)
      new_data<-list(data2)
      data = c(data, new_data)
    }
    pics4 = list(path = path,scantime = scantime[[1]],
                 pics=data)
    pics_1[[j]] = pics4
  }
  return(pics_1)
}
