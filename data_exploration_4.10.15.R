require(data.table)
require(ade4)
require(zoo)
require(ggplot2)
require(lattice)
library(psych)
library(scales)
require(gridExtra)

#load data
setwd("/Users/Lucy/MSDS/2015Spring/DSGA1003_Machine_Learning/project/Clean Datasets")
DT <- fread("Acoustic_data.csv", sep=",",stringsAsFactor = TRUE)
DF <- as.data.frame(DT)
names = names(DT)

for (i in which(sapply(DF[,c(1:37)], is.numeric))) {
  DF[is.na(DF[, i]), i] <- mean(DF[, i],  na.rm = TRUE)
}
write.csv(DF,"acousitc_missing.csv")


DF.normalized <- apply(DF[,-c(1,2,38)], MARGIN = 2, FUN = function (x) (x-min(x))/(max(x)-min(x)))
write.csv(DF.normalized,"acoustic_normalized_temp.csv")



for (i in c(3:38)){
  DT.normalized[,i,with=F] <- (DT[,i,with=F] - min(DT[,i,with=F],na.rm=TRUE)) / (max(DT[,i,with=F],na.rm=TRUE) - min(DT[,i,with=F],na.rm=TRUE))
  
}
DT.case <- fread("Case_data.csv", sep=",",stringsAsFactor = TRUE)

DT <- DT[,V1:=NULL]
DT.case <- DT.case[,V1:=NULL]
DT.case$Lawyer <- as.factor(DT.case$Lawyer)
DT.case$Petitioner <- as.factor(DT.case$Petitioner)
DT.case$Party <- as.factor(DT.case$Party)
DT.case$Year <- as.factor(DT.case$Year)
DT.case$Gender <- as.factor(DT.case$Gender)
DT.case$Segmented <- as.factor(DT.case$Segmented)
DT.case$Role <- as.factor(DT.case$Role)

DT.summary <- as.data.frame(summary(DT.case))
write.csv(DT.summary,"case_summary_4.10.15.csv")

dfplot <- function(data.frame)
{
  df <- data.frame
  ln <- length(names(data.frame))
  for(i in 1:ln){
    mname <- substitute(df[,i])
    if(is.factor(df[,i,with=F])){
      plot(df[,i,with=F],main=names(df)[i])}
    else{hist(df[,i],main=names(df)[i])}
  }
}


#histograms
histograms <- function(DT){
  names <- names(DT)
  par(mfrow=c(3,3))
  end = length(names)-1
  for (i in 1:end){
    ggplot(DT,aes_string(x=names[i])) + geom_density(aes(group=Outcome,colour=as.factor(Outcome),fill=as.factor(Outcome)),alpha=0.3)  
    ggsave(paste(names[i],".pdf"))
  }
}

bars <- function(DT){
  names <- names(DT)
  par(mfrow=c(3,3))
  end = length(names)-1
  for (i in 1:end){
    ggplot(DT,aes_string(x=names[i])) + geom_bar(position="dodge",aes(group=Outcome,colour=as.factor(Outcome),fill=as.factor(Outcome)),alpha=0.3)  
    ggsave(paste(names[i],".pdf"))
  }
}

histograms(DT)
bars(DT.case)

names <- names(DT.case)
ggplot(DT.case,aes_string(x=names[2])) + geom_bar(position="dodge",aes(group=Outcome,colour=as.factor(Outcome),fill=as.factor(Outcome)),alpha=0.3)  




