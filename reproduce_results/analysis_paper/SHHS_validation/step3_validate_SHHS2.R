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
library(stringr)
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
mgh_result_folder <- file.path(base_folder, sprintf('survival_results_%s_AHI%s_MGH_central', sleep_stage_type, with_AHI))
result_path <- file.path(base_folder, sprintf('survival_results_%s_AHI%s2', sleep_stage_type, with_AHI))
if (!dir.exists(result_path))
  dir.create(result_path)

## get data

if (outcome=='Death') {
  dfy <- read_excel(file.path(base_folder, 'SHHS_time2event_IschemicStroke.xlsx'))
} else {
  dfy <- read_excel(file.path(base_folder, sprintf('SHHS_time2event_%s.xlsx', outcome)))
}
if (sleep_stage_type=='5stage') {
  dfX <- read.csv(file.path(base_folder, '../../shared_data/SHHS/to_be_used_features.csv'))
} else {
  dfX <- read.csv(file.path(base_folder, '../../shared_data/SHHS/to_be_used_features_NREM.csv'))
}
#dfX$DateOfVisit <- as.character(dfX$DateOfVisit)
stopifnot(all(dfX$nsrrid==dfy$nsrrid))
dfX <- subset(dfX, select=-c(visitnumber, nsrrid, Race, ESS, AHI, REMPercTST, WASO, TotalSleepTime, SleepEfficiency))
dfy <- subset(dfy, select=-c(nsrrid) )
df <- cbind(dfX, dfy)

# only look at future events relative to sleep study
if (outcome=='Death') {
  ids <- df$time_death>0
} else {
  ids <- (!is.na(df$cens_death))&(df$time_death>0)&(!is.na(df$cens_outcome))&(df$time_outcome>0)
}
df <- df[ids,]

# generate time and event 
if (outcome=='Death') {
  names(df)[names(df)=="time_death"] <- "time"
  names(df)[names(df)=="cens_death"] <- "event"
  df$event <- 1-df$event
  df <- subset(df, select = -c(time_outcome, cens_outcome) )
} else {
  df$event2_occur = 1-df$cens_death
  df$event1_occur = 1-df$cens_outcome
  df$time <- with(df, ifelse(event1_occur==0, time_death, time_outcome))
  event <- with(df, ifelse(event1_occur==0, 2*event2_occur, 1))
  df$event <- factor(event, 0:2, labels=c("censor", "event1", "event2"))
  df <- subset(df, select = -c(time_death, time_outcome, cens_death, cens_outcome, event1_occur, event2_occur) )
}
#df$id <- 1:nrow(df)
rownames(df) <- NULL
ynames <- c('time', 'event')

## load Xmean and Xstd

mat <- readMat(file.path(mgh_result_folder, sprintf('results_%s_%s.mat', outcome, model_type)))
Xtr_before_impute <- mat$Xtr
Xnames <- unlist(mat$Xmean.names)
Xmean <- mat$Xmean
Xstd <- mat$Xstd
names(Xmean) <- Xnames
names(Xstd) <- Xnames
# only take intersect between MGH and SHHS features
Xnames <- intersect(Xnames, names(df))
Xmean <- Xmean[Xnames]
Xstd  <- Xstd[Xnames]

# preprocessing: standardization and imputation
X <- as.matrix(df[,Xnames])
# log-transform
#X <- sign(X)*log1p(abs(X))
# normalize
X_before_impute <- (X -t(replicate(nrow(X), Xmean)) ) / t(replicate(nrow(X), Xstd))
sklearn.impute <- import("sklearn.impute")
knn.model <- sklearn.impute$KNNImputer(n_neighbors=10L)
knn.model$fit(Xtr_before_impute)
df[,Xnames] <- knn.model$transform(X_before_impute)

#df <- df[1:300,]
print(dim(df))

# load model
model <- readRDS(file.path(mgh_result_folder, sprintf('model_%s_%s.rda', outcome, model_type)))
model <- model[[1]] # only 1 channel, no need to average

coef <- model$coefficients
coef <- coef[1:(length(coef)/2)]
names(coef) <- str_replace_all(names(coef), "_1:2", "")

# compute zp
covariate_names <- c('Age', 'Sex', 'BMI', 'MedBenzo', 'MedAntiDep', 'MedStimulant')#, 'MedSedative', 'MedAntiEplipetic'
age_mean <- 50#mean(df$Age)
sex_mean <- 0.5#mean(df$Sex)
bmi_mean <- 30#mean(df$BMI)
benzo <- 0
antidep <- 0
#sedative <- 0
#antieplipetic <- 0
stimulant <- 0
actual.sex <- df$Sex
rep_val <- c(age_mean, sex_mean, bmi_mean, benzo, antidep, stimulant)#, sedative, antieplipetic
df[,covariate_names] <- t(replicate(nrow(df), (rep_val-Xmean[covariate_names])/Xstd[covariate_names]))
zp <- drop(as.matrix(df[,names(coef)])%*%coef)

levels <- c(1,2,3)
sexs   <- c(0,1)

# get Cox estimate
print('survfit starts')
st <- Sys.time()
res <- survfit(model, df)
res2 <- summary(res)
et <- Sys.time()
print('survfit ends')
print(et-st)
cox_survtime <- res2$time
if (model_type=='CoxPH') {
  cox_survstate <- c('event1')
} else {
  cox_survstate <- res2$states
}

# start bootstrapping
cox_curves_bt <- list()
for (bti in 1:(Nbt+1)) {
  if (bti%%10==0)
    print(bti)
  # generate df_bt
  if (bti==1) {
    df_bt <- copy(df)
    zp_bt <- zp
    actual.sex.bt <- actual.sex
    if (model_type=='CoxPH') {
      survprob_bt <- res2$surv
    } else {
      survprob_bt <- res2$pstate
    }
  } else {
    set.seed(random_seed+bti)
    bt_ids <- sample(1:nrow(df), replace=TRUE) # sample with replacement
    df_bt <- copy(df)
    df_bt <- df_bt[bt_ids,]
    zp_bt <- zp[bt_ids]
    actual.sex.bt <- actual.sex[bt_ids]
    if (model_type=='CoxPH') {
      survprob_bt <- res2$surv[,bt_ids]
    } else {
      survprob_bt <- res2$pstate[,bt_ids,]
    }
  }
  df_bt$id <- 1:nrow(df_bt)
  
  z_boundaries <- quantile(zp_bt, probs=c(0.25,0.75))
  zp2 <- rep(2, nrow(df_bt))
  zp2[zp_bt<z_boundaries[1]] <- 1
  zp2[zp_bt>z_boundaries[2]] <- 3
  
  df2 <- copy(df_bt)
  df2$zp_factor  <- factor(zp2, levels)
  df2$actual.sex <- factor(actual.sex.bt, sexs)
  
  if (bti==1) {
    # get AJ estimate
    AJ_fit <- survfit(Surv(time,event)~zp_factor+actual.sex, data=df2, id=id)
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
  }
  
  # get Cox estimate
  cox_curves <- list()
  for (sex in sexs) {
    for (level in levels) {
      #print(sprintf('level = %d, sex = %d', level, sex))
      ids <- (df2$zp_factor==level)&(df2$actual.sex==sex)
      if (sum(ids)==0) {
        next
      }
      if (model_type=='CoxPH') {
        cox_survprob <- 1-apply(survprob_bt[,ids,drop=F],1,mean)
        # reshape
        dim1 <- length(cox_survprob)
        cox_survprob <- array(cox_survprob, dim=c(dim1, 1))
      } else {
        cox_survprob <- apply(survprob_bt[,ids,,drop=F],c(1,3),mean)
        # reshape
        dim1 <- dim(cox_survprob)[1]
        dim2 <- dim(cox_survprob)[2]
        cox_survprob <- array(cox_survprob, dim=c(dim1, 1, dim2))
      }
      cox_curves[[length(cox_curves)+1]] <- cox_survprob
    }
  }
  cox_curves <- simplify2array(cox_curves)
  if (model_type=='CoxPH') {
    cox_curves <- cox_curves[,1,]
  } else {
    cox_curves <- aperm(cox_curves, c(1,2,4,3))
    cox_curves <- cox_curves[,1,,]
  }
  cox_curves_bt[[length(cox_curves_bt)+1]] <- cox_curves
}
cox_curves_bt <- simplify2array(cox_curves_bt)
if (model_type=='CoxPH') {
  cox_curves_bt <- aperm(cox_curves_bt, c(1,3,2))
} else {
  cox_curves_bt <- aperm(cox_curves_bt, c(1,4,2,3))
}

writeMat(file.path(result_path, sprintf('results_%s_%s.mat', outcome, model_type)),
         cox_curve_tes_bt_time=cox_survtime,
         cox_curve_tes_bt_val=cox_curves_bt,
         cox_curve_tes_bt_states=cox_survstate,
         AJ_curve_tes_time=AJ_survtime,
         AJ_curve_tes_val=AJ_survprob,
         AJ_curve_tes_states=AJ_survstate,
         AJ_curve_tes_upper=AJ_upper,
         AJ_curve_tes_lower=AJ_lower)
