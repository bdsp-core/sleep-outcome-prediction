#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)

## load packages

library(survival)
library(R.matlab)
library(doParallel)
library(readxl)
library(data.table)  # for copy data.frame
#library(impute)
library(reticulate)  # for sklearn.impute.KNNImputer
source('../Rfunctions.R')

## set up parameters

base_folder <- '/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction (1)/code-haoqi/SHHS_validation'
Nbt <- 1000
random_seed <- 2021

outcome <- args[1]
if (outcome=='Death') {
  model_type <- 'CoxPH'
} else {
  model_type <- 'CoxPH_CompetingRisk'
}
sleep_stage_type <- args[2] #'baseline', '5stage', 'NREM'
#if (length(args)>=3) {
#  with_AHI <- args[3]=='with_AHI' #with_AHI, without_AHI
#} else {
with_AHI <- FALSE
#}
mgh_result_folder <- sprintf('/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction (1)/code-haoqi/survival_results_%s_AHI%s', sleep_stage_type, with_AHI)
mgh_time2event_folder <- '/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction (1)/code-haoqi'

result_path <- file.path(base_folder, sprintf('survival_results_%s_AHI%s', sleep_stage_type, with_AHI))
if (!dir.exists(result_path))
  dir.create(result_path)

## get data

if (outcome=='Death') {
  dfy <- read_excel(file.path(base_folder, 'SHHS_time2event_IschemicStroke.xlsx'))
  dftry <- read_excel(file.path(mgh_time2event_folder, 'time2event_IschemicStroke.xlsx'))
} else {
  dfy <- read_excel(file.path(base_folder, sprintf('SHHS_time2event_%s.xlsx', outcome)))
  dftry <- read_excel(file.path(mgh_time2event_folder, sprintf('time2event_%s.xlsx', outcome)))
}
if (sleep_stage_type=='5stage') {
  dfX <- read.csv(file.path(base_folder, '../../shared_data/SHHS/to_be_used_features.csv'))
  dftrX <- read.csv(file.path(base_folder, '../../shared_data/MGH/to_be_used_features.csv'))
} else {
  dfX <- read.csv(file.path(base_folder, '../../shared_data/SHHS/to_be_used_features_NREM.csv'))
  dftrX <- read.csv(file.path(base_folder, '../../shared_data/MGH/to_be_used_features_NREM.csv'))
}
#dfX$DateOfVisit <- as.character(dfX$DateOfVisit)
stopifnot(all(dfX$nsrrid==dfy$nsrrid))
dfX <- subset(dfX, select=-c(visitnumber, nsrrid, Race, ESS, AHI, REMPercTST, WASO, TotalSleepTime, SleepEfficiency))
dfy <- subset(dfy, select=-c(nsrrid) )
df <- cbind(dfX, dfy)
stopifnot(all(dftrX$MRN==dftry$MRN))
dftry <- subset(dftry, select=-c(PatientID, MRN, DateOfVisit))
dftr <- cbind(dftrX, dftry)
  
# only look at future events relative to sleep study
if (outcome=='Death') {
  ids <- df$time_death>0
  idstr <- dftr$time_death>0
} else {
  ids <- (!is.na(df$cens_death))&(df$time_death>0)&(!is.na(df$cens_outcome))&(df$time_outcome>0)
  idstr <- (!is.na(df$cens_death))&(df$time_death>0)&(!is.na(df$cens_outcome))&(df$time_outcome>0)
}
df <- df[ids,]
dftr <- dftr[idstr,]

# generate time and event 
if (outcome=='Death') {
  names(df)[names(df)=="time_death"] <- "time"
  names(df)[names(df)=="cens_death"] <- "event"
  df$event <- 1-df$event
  df <- subset(df, select = -c(time_outcome, cens_outcome) )
  
  names(dftr)[names(dftr)=="time_death"] <- "time"
  names(dftr)[names(dftr)=="cens_death"] <- "event"
  dftr$event <- 1-dftr$event
  dftr <- subset(dftr, select = -c(time_outcome, cens_outcome) )
} else {
  df$event2_occur = 1-df$cens_death
  df$event1_occur = 1-df$cens_outcome
  df$time <- with(df, ifelse(event1_occur==0, time_death, time_outcome))
  event <- with(df, ifelse(event1_occur==0, 2*event2_occur, 1))
  df$event <- factor(event, 0:2, labels=c("censor", "event1", "event2"))
  df <- subset(df, select = -c(time_death, time_outcome, cens_death, cens_outcome, event1_occur, event2_occur) )
  
  dftr$event2_occur = 1-dftr$cens_death
  dftr$event1_occur = 1-dftr$cens_outcome
  dftr$time <- with(dftr, ifelse(event1_occur==0, time_death, time_outcome))
  event <- with(dftr, ifelse(event1_occur==0, 2*event2_occur, 1))
  dftr$event <- factor(event, 0:2, labels=c("censor", "event1", "event2"))
  dftr <- subset(dftr, select = -c(time_death, time_outcome, cens_death, cens_outcome, event1_occur, event2_occur) )
}
df$id <- 1:nrow(df)
dftr$id <- 1:nrow(dftr)
rownames(df) <- NULL
rownames(dftr) <- NULL
ynames <- c('time', 'event')

## load Xmean and Xstd

mat <- readMat(file.path(mgh_result_folder, sprintf('results_%s_%s.mat', outcome, model_type)))
Xnames_all <- unlist(mat$Xmean.names)
Xmean <- mat$Xmean
Xstd <- mat$Xstd
names(Xmean) <- Xnames_all
names(Xstd) <- Xnames_all
# only take intersect between MGH and SHHS features
Xnames_all <- intersect(Xnames_all, names(df))
Xmean <- Xmean[Xnames_all]
Xstd  <- Xstd[Xnames_all]

# preprocessing: standardization and imputation
Xtr <- as.matrix(dftr[,Xnames_all])
X <- as.matrix(df[,Xnames_all])
# log-transform
#X <- sign(X)*log1p(abs(X))
# normalize
Xtr_before_impute <- (Xtr -t(replicate(nrow(Xtr), Xmean)) ) / t(replicate(nrow(Xtr), Xstd))
X_before_impute <- (X -t(replicate(nrow(X), Xmean)) ) / t(replicate(nrow(X), Xstd))
sklearn.impute <- import("sklearn.impute")
knn.model <- sklearn.impute$KNNImputer(n_neighbors=10L)
knn.model$fit(Xtr_before_impute)
dftr[,Xnames_all] <- knn.model$transform(Xtr_before_impute)
df[,Xnames_all] <- knn.model$transform(X_before_impute)

print(dim(df))

## load coef
coef <- read.csv(file.path(mgh_result_folder, sprintf('coef_%s_%s.csv', outcome, model_type)))
if (model_type=='CoxPH') {
  rownames(coef) <- coef$X
} else {
  coef <- coef[1:(nrow(coef)/2),]
  rownames(coef) <- gsub('_1:2','',coef$X)
}
Xnames <- intersect(Xnames_all, rownames(coef))
coef <- coef[Xnames,]$coef

## find average age and sex
age_mean <- 50#mean(df$Age)
sex_mean <- 0.5#mean(df$Sex)
bmi_mean <- 30#mean(df$BMI)
benzo <- 0
antidep <- 0
#sedative <- 0
#antieplipetic <- 0
stimulant <- 0

## get boundaries

actual.sex.tr <- dftr$Sex
dftr$Age          <- (age_mean-Xmean['Age'])/Xstd['Age']
dftr$Sex          <- (sex_mean-Xmean['Sex'])/Xstd['Sex']
dftr$BMI          <- (bmi_mean-Xmean['BMI'])/Xstd['BMI']
dftr$MedBenzo     <- (benzo-Xmean['MedBenzo'])/Xstd['MedBenzo']
dftr$MedAntiDep   <- (antidep-Xmean['MedAntiDep'])/Xstd['MedAntiDep']
dftr$MedStimulant <- (stimulant-Xmean['MedStimulant'])/Xstd['MedStimulant']
zptr <- drop(as.matrix(dftr[,Xnames])%*%coef)

# load MGH_match_mask.npy
#np <- import("numpy")
#MGH_match_mask <- np$load("MGH_match_mask.npy")
#dftr <- dftr[MGH_match_mask,]
#zptr <- zptr[MGH_match_mask]
#actual.sex.tr <- actual.sex.tr[MGH_match_mask]

levels <- c(1,2,3)
sex_levels <- c(0,1)

cox_curve_tes_bt <- list()
AJ_curve_tes  <- list()
set.seed(random_seed)
for (bti in 1:(Nbt+1)) {
  if (bti%%10==0)
    print(bti)
  if (bti==1) {
    df_bt <- copy(df)
  } else {
    set.seed(random_seed+bti)
    bt_ids <- sample(1:nrow(df), replace=TRUE) # sample with replacement
    df_bt <- copy(df)
    df_bt <- df_bt[bt_ids,]
  }
  df_bt$id <- 1:nrow(df_bt)

  actual.sex <- df_bt$Sex
  df_bt$Age          <- (age_mean-Xmean['Age'])/Xstd['Age']
  df_bt$Sex          <- (sex_mean-Xmean['Sex'])/Xstd['Sex']
  df_bt$BMI          <- (bmi_mean-Xmean['BMI'])/Xstd['BMI']
  df_bt$MedBenzo     <- (benzo-Xmean['MedBenzo'])/Xstd['MedBenzo']
  df_bt$MedAntiDep   <- (antidep-Xmean['MedAntiDep'])/Xstd['MedAntiDep']
  df_bt$MedStimulant <- (stimulant-Xmean['MedStimulant'])/Xstd['MedStimulant']
  zpte <- drop(as.matrix(df_bt[,Xnames])%*%coef)
  z_boundaries <- quantile(zpte, probs=c(0.25,0.75))  # use boundaries based on training set
  
  zpte2 <- rep(2, length(zpte))
  zpte2[zpte<z_boundaries[1]] <- 1
  zpte2[zpte>z_boundaries[2]] <- 3
  df_bt$zp_factor <- factor(zpte2, levels)
  df_bt$actual.sex <- factor(actual.sex, sex_levels)
  
  zptr2 <- rep(2, nrow(dftr))
  zptr2[zptr<z_boundaries[1]] <- 1
  zptr2[zptr>z_boundaries[2]] <- 3
  dftr$zp_factor <- factor(zptr2, levels)
  dftr$actual.sex <- factor(actual.sex.tr, sex_levels)
  cox_fit <- coxph(Surv(time,event)~zp_factor + actual.sex, data=dftr, id=id, ties='breslow')
  cox_res <- survfit(cox_fit, data.frame(
    zp_factor=factor(c(levels,levels), levels=levels),
    actual.sex=factor(c(0,0,0,1,1,1), levels=sex_levels)))
  cox_res <- summary(cox_res)
  cox_survtime <- cox_res$time
  if (model_type=='CoxPH') {
    cox_survprob <- 1-cox_res$surv
    cox_survstate <- c('event1')
  } else {
    cox_survprob <- cox_res$pstate
    cox_survstate <- cox_res$states
  }
  cox_curve_tes_bt[[length(cox_curve_tes_bt)+1]] <- list(time=cox_survtime, val=cox_survprob, state=cox_survstate)
  
  # get AJ estimates only for bti==1 because it has its own CI estimate
  if (bti==1) {
    AJ_fit <- survfit(Surv(time,event)~zp_factor+actual.sex, data=df_bt, id=id)
    AJ_res <- summary(AJ_fit)
    AJ_res <- harmonize.curves.AJ(AJ_res, c('zp_factor=1, actual.sex=0', 'zp_factor=2, actual.sex=0', 'zp_factor=3, actual.sex=0', 'zp_factor=1, actual.sex=1', 'zp_factor=2, actual.sex=1', 'zp_factor=3, actual.sex=1'))
    AJ_survtime <- AJ_res$time
    if (model_type=='CoxPH') {
      AJ_survprob <- 1-AJ_res$surv
      AJ_survstate <- c('event1')
      AJ_upper <- 1-AJ_res$lower
      AJ_lower <- 1-AJ_res$upper
    } else {
      AJ_survprob <- AJ_res$pstate
      AJ_survstate <- AJ_res$states
      AJ_upper <- AJ_res$upper
      AJ_lower <- AJ_res$lower
    }
    AJ_curve_tes <- list(time=AJ_survtime, val=AJ_survprob, state=AJ_survstate, lower=AJ_lower, upper=AJ_upper)
  }
}
cox_curve_tes_bt <- harmonize.curves2(cox_curve_tes_bt)

writeMat(file.path(result_path, sprintf('results_%s_%s.mat', outcome, model_type)),
         z_boundaries=z_boundaries,
         cox_curve_tes_bt_time=cox_curve_tes_bt$time,
         cox_curve_tes_bt_val=cox_curve_tes_bt$val,
         cox_curve_tes_bt_states=cox_curve_tes_bt$state,
         AJ_curve_tes_time=AJ_curve_tes$time,
         AJ_curve_tes_val=AJ_curve_tes$val,
         AJ_curve_tes_states=AJ_curve_tes$state,
         AJ_curve_tes_upper=AJ_curve_tes$upper,
         AJ_curve_tes_lower=AJ_curve_tes$lower)
