#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)

## load packages

library(plyr)
library(survival)
library(R.matlab)
library(glmnet)
library(groupdata2)  # for fold
library(doParallel)
library(foreach)
library(abind)
library(readxl)
library(timeROC)
#library(survcomp)
library(data.table)  # for copy data.frame
#library(impute)
library(reticulate)  # for sklearn.impute.KNNImputer
source('Rfunctions.R')

## set up parameters

base_folder <- '/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction (1)/code-haoqi'
Ncv <- 5
Nbt <- 1000
random_seed <- 2021
alpha_list <- c(0.5,0.6,0.7,0.8,0.9)

outcome <- args[1]
exposure_type <- args[2] #'AHI', '5stage', 'NREM'
if (length(args)>=3) {
  sensitivity_analysis <- args[3] #'sens1', 'sens2'
} else {
  sensitivity_analysis <- 'none'
}
stopifnot(sensitivity_analysis %in% c('sens1', 'sens2', 'none'))#, 'male', 'female'
stopifnot((sensitivity_analysis!='sens1')|(sensitivity_analysis!='sens2')|(outcome!='Death')) # no sensitivity analysis for outcome=Death, since this is competing risk only
if ((outcome=='Death')|(sensitivity_analysis=='sens1')|(sensitivity_analysis=='sens2')) {
  model_type <- 'CoxPH'
} else {
  model_type <- 'CoxPH_CompetingRisk'
}
if (sensitivity_analysis=='none') {
  result_path <- file.path(base_folder, sprintf('survival_results_%s_bt%d', exposure_type, Nbt))
} else {
  result_path <- file.path(base_folder, sprintf('survival_results_%s_bt%d_%s', exposure_type, Nbt, sensitivity_analysis))
}
if (!dir.exists(result_path))
  dir.create(result_path)

## get data

if (outcome=='Death') {
  dfy <- read_excel(file.path(base_folder, 'time2event_IschemicStroke.xlsx'))
} else {
  dfy <- read_excel(file.path(base_folder, sprintf('time2event_%s.xlsx', outcome)))
}
if (exposure_type=='5stage') {
  dfX <- read.csv(file.path(base_folder, '../shared_data/MGH/to_be_used_features.csv'))
} else {
  dfX <- read.csv(file.path(base_folder, '../shared_data/MGH/to_be_used_features_NREM.csv'))
}
dfX$DateOfVisit <- as.character(dfX$DateOfVisit)
stopifnot(all(dfX$MRN==dfy$MRN))
stopifnot(all(dfX$DateOfVisit==dfy$DateOfVisit))
dfX <- subset(dfX, select=-c(DateOfVisit, TwinDataID, Path) )
dfy <- subset(dfy, select=-c(PatientID, MRN, DateOfVisit) )
df <- cbind(dfX, dfy)

# get outer CV ids
df_cv <- read.csv(file.path(base_folder, 'outer_cv_foldid.csv'))
stopifnot(all(df$MRN==df_cv$MRN))
stopifnot(all(df$DateOfVisit==df_cv$DateOfVisit))
df$CVFold <- df_cv$CVFold

# only look at future events relative to sleep study
if (outcome=='Death') {
  ids <- df$time_death>0
} else {
  ids <- (!is.na(df$cens_death))&(df$time_death>0)&(!is.na(df$cens_outcome))&(df$time_outcome>0)
}
df <- df[ids,]

#if (sensitivity_analysis=='male') {
#  df <- df[df$Sex==1,]
#  df <- subset(df, select=-Sex)
#} else if (sensitivity_analysis=='female') {
#  df <- df[df$Sex==0,]
#  df <- subset(df, select=-Sex)
#}
row.names(df) <- NULL

# generate time and event 

if (outcome=='Death') {
  names(df)[names(df)=="time_death"] <- "time"
  names(df)[names(df)=="cens_death"] <- "event"
  df$event <- 1-df$event
  df <- subset(df, select = -c(time_outcome, cens_outcome) )
} else if (sensitivity_analysis=='sens1') {
  # all subjects that are dead are assumed to have the outcome instead
  df$cens_outcome[(df$cens_outcome==1)&(df$cens_death==0)] <- 0
  names(df)[names(df)=="time_outcome"] <- "time"
  names(df)[names(df)=="cens_outcome"] <- "event"
  df$event <- 1-df$event
  df <- subset(df, select = -c(time_death, cens_death) )
} else if (sensitivity_analysis=='sens2') {
  # all subjects that are dead are assumed to not have the outcome for as long as the largest survival time
  longest_survival_time <- max(max(df$time_outcome), max(df$time_death))
  print(sprintf('Longest survival time: %f yr', longest_survival_time))
  df$time_outcome[(df$cens_outcome==1)&(df$cens_death==0)] <- longest_survival_time
  names(df)[names(df)=="time_outcome"] <- "time"
  names(df)[names(df)=="cens_outcome"] <- "event"
  df$event <- 1-df$event
  df <- subset(df, select = -c(time_death, cens_death) )
} else {
  df$event2_occur <- 1-df$cens_death
  df$event1_occur <- 1-df$cens_outcome
  df$time <- with(df, ifelse(event1_occur==0, time_death, time_outcome))
  event <- with(df, ifelse(event1_occur==0, 2*event2_occur, 1))
  df$event <- factor(event, 0:2, labels=c("censor", "event1", "event2"))
  df <- subset(df, select = -c(time_death, time_outcome, cens_death, cens_outcome, event1_occur, event2_occur) )
}
df$id <- 1:nrow(df)
rownames(df) <- NULL
ynames <- c('time', 'event')

## now data loading is done, print data shape

print(dim(df))

## get average age and sex to be used later

age_mean <- 50#mean(df$Age)
sex_mean <- 0.5#mean(df$Sex)
bmi_mean <- 30#mean(df$BMI)
benzo <- 0
antidep <- 0
sedative <- 0
antieplipetic <- 0
stimulant <- 0
#if ((sensitivity_analysis=='male')|(sensitivity_analysis=='female')) {
#  cov_fix_val <- c(age_mean, bmi_mean, benzo, antidep, sedative, antieplipetic, stimulant)
#} else {
cov_fix_val <- c(age_mean, sex_mean, bmi_mean, benzo, antidep, sedative, antieplipetic, stimulant)
#}

## do an overall AJ curve

AJ_fit <- survfit(Surv(time, event) ~ 1, data=df, id=id)
AJ_fit <- summary(AJ_fit)
save_path <- file.path(result_path, sprintf('AJ_output_%s_%s_overall.mat', outcome, exposure_type))
if (model_type=='CoxPH_CompetingRisk') {
  writeMat(save_path,
           time=AJ_fit$time, val=AJ_fit$pstate, lower=AJ_fit$lower, upper=AJ_fit$upper, states=AJ_fit$states
  )
} else {
  # for single outcome, death, AJ = 1-KM
  writeMat(save_path,
           time=AJ_fit$time, val=1-AJ_fit$surv, lower=1-AJ_fit$upper, upper=1-AJ_fit$lower, states=c('event1')
  )
}

## determine input features (X names) and y names
covariate_names <- c('Age', 'Sex', 'BMI', 'MedBenzo', 'MedAntiDep', 'MedSedative', 'MedAntiEplipetic', 'MedStimulant')
#if ((sensitivity_analysis=='male')|(sensitivity_analysis=='female')) {
#  covariate_names <- covariate_names[covariate_names!='Sex']
#}
covariate.is.discrete <- c(F, T, F, T, T, T, T, T)
if (exposure_type!='AHI') {
  remove_names <- c(ynames, 'MRN', 'PatientID', 'CVFold', 'id', covariate_names, 'AHI')
  remove_names <- c(remove_names, 'NREMPercTST', 'N2PercTST', 'BMI', 'TotalSleepTime', 'SleepEfficiency', 'REMPercTST', 'WASO')
  remove_names <- c(remove_names,  Filter(function(x) grepl('(_W$|_W_)',x), names(df)))
  remove_names <- c(remove_names,  Filter(function(x) grepl('(theta|delta)_bandpower_mean_(F|C|O)_(NREM|R)',x), names(df)))
  remove_names <- c(remove_names,  Filter(function(x) grepl('SO_AMP_(F|C|O)',x), names(df)))
  remove_names <- c(remove_names,  Filter(function(x) grepl('SO_POS_DUR_(F|C|O)',x), names(df)))
  remove_names <- c(remove_names,  Filter(function(x) grepl('SO_NEG_DUR_(F|C|O)',x), names(df)))
  remove_names <- c(remove_names,  Filter(function(x) grepl('delta_theta_mean_(F|C|O)_(NREM|R)',x), names(df)))
  xnames <- names(df)[!(names(df)%in%remove_names)]
} else {
  xnames <- c('AHI')
}

# get feature names for F, C, O brain regions
xnames_mask_f <- sapply(xnames, function(x) grepl('(_F$|_F_)',x))
xnames_mask_c <- sapply(xnames, function(x) grepl('(_C$|_C_)',x))
xnames_mask_o <- sapply(xnames, function(x) grepl('(_O$|_O_)',x))
xnames_f <- xnames[xnames_mask_f]
xnames_c <- xnames[xnames_mask_c]
xnames_o <- xnames[xnames_mask_o]

## cross validation with bootstrapping

cindex_tes_bt <- c()
#auc_tes_bt <- list()
cox_curve_tes_bt <- list()
sklearn.impute <- import("sklearn.impute")
set.seed(random_seed)

for (bti in 1:(Nbt+1)) {
  if (bti%%10==0)
    print(bti)
  # generate df_bt
  if (bti==1) {
    df_bt <- copy(df)
    best_alphas <- c()
  } else {
    set.seed(random_seed+bti)
    bt_ids <- sample(1:nrow(df), replace=TRUE) # sample with replacement
    df_bt <- copy(df)
    df_bt <- df_bt[bt_ids,]
  }
  df_bt$id <- 1:nrow(df_bt)

  tryCatch(  # some bootstrap will throw error, if so, ignore and continue
    expr = {
      # cross validation
      cindex_tes <- c()
      #auc_tes <- list()
      cox_curve_tes <- list()
      if (bti==1)
        AJ_curve_tes <- list()
      for (cvi in 1:Ncv) { # outer loop
        dftr <- df_bt[df_bt$CVFold!=cvi,]
        dfte <- df_bt[df_bt$CVFold==cvi,]
        
        Xtr <- as.matrix(dftr[,c(xnames, covariate_names)])
        # log-transform
        #Xtr <- sign(Xtr)*log1p(abs(Xtr))
        # normalize
        Xmean <- apply(Xtr, 2, mean, na.rm=TRUE)
        Xmean[c(rep(F,length(xnames)), covariate.is.discrete)] <- 0
        Xstd <- apply(Xtr, 2, sd, na.rm=TRUE)
        Xstd[c(rep(F,length(xnames)), covariate.is.discrete)] <- 1
        Xmean_covariate_names <- Xmean[(length(xnames)+1):length(Xmean)]
        Xstd_covariate_names <- Xstd[(length(xnames)+1):length(Xmean)]
        Xtr_before_impute <- ( Xtr-t(replicate(nrow(Xtr), Xmean)) ) / t(replicate(nrow(Xtr), Xstd))
        
        # impute missing value
        #Xtr <- impute.knn(Xtr_before_impute, k=10)$data
        knn.model <- sklearn.impute$KNNImputer(n_neighbors=10L)
        knn.model$fit(Xtr_before_impute)
        Xtr <- knn.model$transform(Xtr_before_impute)
        dftr[,c(xnames,covariate_names)] <- Xtr
        
        if (bti==1) {
          # fit a Cox ElasticNet model to reduce dimension 
          if (exposure_type!='AHI') {
            Xtr <- as.matrix(dftr[,xnames])
            # ElasticNet is only done for the outcome
            if (model_type=='CoxPH') {
              ytr <- Surv(dftr$time, dftr$event==1)
            } else {
              ytr <- Surv(dftr$time, dftr$event=='event1')
            }
            foldid <- as.numeric(fold(as.data.frame(1:nrow(dftr)), Ncv)$.folds) # generate inner loop id
            best_cvm <- -Inf
            for (alpha_ in alpha_list) {
              print(alpha_)
              coxph_cv_fit <- fit_cv_cox_model(Xtr, ytr, list(xnames_f, xnames_c, xnames_o), foldid, alpha_, n.core=Ncv)
              cvm <- get_cvm(coxph_cv_fit)
              if (cvm>best_cvm) {
                best_cvm <- cvm
                best_alpha_outer_loop <- alpha_
                best_coxph_cv_fit <- coxph_cv_fit
              }
            }
            print(best_alpha_outer_loop)
            coef_ <- get_cv_cox_coef(best_coxph_cv_fit, list(xnames_mask_f, xnames_mask_c, xnames_mask_o))
            xnames2 <- c(xnames[abs(coef_)>0], covariate_names)
          } else {
            best_alpha_outer_loop <- -1
            xnames2 <- c(xnames, covariate_names)
          }
          best_alphas <- c(best_alphas, best_alpha_outer_loop)
          
          # fit competing risk model with selected features
          xnames2_mask_f <- sapply(xnames2, function(x) grepl('(_F$|_F_)',x)|(x%in%covariate_names))
          xnames2_mask_c <- sapply(xnames2, function(x) grepl('(_C$|_C_)',x)|(x%in%covariate_names))
          xnames2_mask_o <- sapply(xnames2, function(x) grepl('(_O$|_O_)',x)|(x%in%covariate_names))
          xnames2_f <- xnames2[xnames2_mask_f]
          xnames2_c <- xnames2[xnames2_mask_c]
          xnames2_o <- xnames2[xnames2_mask_o]
        }
        
        if (model_type=='CoxPH') {
          fit_model <- fit_cox_model
          get_coef  <- get_cox_coef
        } else {
          fit_model <- fit_competing_risk_model
          get_coef  <- get_competing_risk_coef
        }
        if (exposure_type!='AHI') {
          coxph_fit <- fit_model(dftr, list(xnames2_f, xnames2_c, xnames2_o))
          coxph_coef <- get_coef(coxph_fit, xnames2)
        } else {
          coxph_fit <- fit_model(dftr, list(xnames2))
          model_summary <- summary(coxph_fit[[1]])
          coxph_coef <-cbind(model_summary$coefficients, model_summary$conf.int)
          coxph_coef <- coxph_coef[, c('coef', 'Pr(>|z|)', 'lower .95', 'upper .95')]
        }
        
        # test on testing set
        Xte <- as.matrix(dfte[,c(xnames,covariate_names)])
        #Xte <- sign(Xte)*log1p(abs(Xte))
        Xte <- ( Xte-t(replicate(nrow(Xte), Xmean)) ) / t(replicate(nrow(Xte), Xstd))
        #Xte <- my.impute.knn.test(Xte, k=10, Xtr=Xtr) # find nearest neighbor inside training data
        Xte <- knn.model$transform(Xte)
        dfte[,c(xnames,covariate_names)] <- Xte
        
        # get testing performances: C-index
        if (model_type=='CoxPH_CompetingRisk') {
          zpte1 <- drop(as.matrix(dfte[,xnames2])%*%coxph_coef[1:(nrow(coxph_coef)/2),'coef'])
          zpte2 <- drop(as.matrix(dfte[,xnames2])%*%coxph_coef[(nrow(coxph_coef)/2+1):nrow(coxph_coef),'coef'])
          names(zpte1) <- NULL
          names(zpte2) <- NULL
          
          # reproduce coxph.fit$concordance
          #cindex_te <- get_concordance(dfte[ynames], zpte)
          #cindex_te <- concordancefit(Surv(dfte$time,dfte$event),zpte,reverse=TRUE)$concordance
          y2 <- Surv(dfte$time, dfte$event)
          y2 <- aeqSurv(y2)
          y2 <- Surv(c(y2[,1], y2[,1]),c(as.integer(y2[,2]==1), as.integer(y2[,2]==2)))
          zpte <- c(zpte1, zpte2)
          istrat <- c(rep(1, length(zpte1)), rep(2,length(zpte2)))
          res <- concordancefit(y2, zpte, istrat, reverse=TRUE, timefix=FALSE)
          #cindex_te <- c("C"=res$concordance, "se(C)"=sqrt(res$var))
          cindex_te <- res$concordance
        } else {
          zpte1 <- drop(as.matrix(dfte[,xnames2])%*%coxph_coef[,'coef'])
          names(zpte1) <- NULL
          zpte2 <- c()
          
          # reproduce coxph.fit$concordance
          y2 <- Surv(dfte$time, dfte$event)
          y2 <- aeqSurv(y2)
          res <- concordancefit(y2, zpte1, reverse=TRUE, timefix=FALSE)
          #cindex_te <- c("C"=res$concordance, "se(C)"=sqrt(res$var))
          cindex_te <- res$concordance
        }
        cindex_tes <- c(cindex_tes, cindex_te)
        
        # get testing performances: AUC
        #if (model_type=='CoxPH') {
        #  auc_res <- timeROC(T=dfte$time,delta=as.integer(dfte$event),marker=zpte1,times=2:10, cause=1, iid=TRUE)
        #  auc_te  <- auc_res$AUC
        #} else {
        #  auc_res <- timeROC(T=dfte$time,delta=as.integer(dfte$event),marker=zpte1,times=2:10, cause=2, iid=TRUE)
        #  #auc_te  <- cbind(point=auc_res$AUC_1, confint(auc_res)$CI_AUC_1/100)
        #  auc_te  <- auc_res$AUC_1
        #}
        #auc_tes[[length(auc_tes)+1]] <- auc_te
        
        # get testing performances: cumulative incidence curves
        actual.sex.tr <- dftr$Sex
        actual.sex.te <- dfte$Sex
        dftr[,covariate_names] <- t(replicate(nrow(dftr), (cov_fix_val-Xmean_covariate_names)/Xstd_covariate_names))
        dfte[,covariate_names] <- t(replicate(nrow(dfte), (cov_fix_val-Xmean_covariate_names)/Xstd_covariate_names))
        
        if (model_type=='CoxPH') {
          zptr <- drop(as.matrix(dftr[,xnames2])%*%coxph_coef[,'coef'])
          zpte <- drop(as.matrix(dfte[,xnames2])%*%coxph_coef[,'coef'])
        } else{
          zptr <- drop(as.matrix(dftr[,xnames2])%*%coxph_coef[1:(nrow(coxph_coef)/2),'coef'])
          zpte <- drop(as.matrix(dfte[,xnames2])%*%coxph_coef[1:(nrow(coxph_coef)/2),'coef'])
        }
        
        sex_levels <- c(0,1)
        if (exposure_type=='AHI') {
          levels <- c(1,2,3)
          AHI_boundaries <- c(15,30)
          zptr2 <- rep(2, nrow(dftr))
          zptr2[dftr$AHI<(AHI_boundaries[1]-Xmean[['AHI']])/Xstd[['AHI']]] <- 1
          zptr2[dftr$AHI>(AHI_boundaries[2]-Xmean[['AHI']])/Xstd[['AHI']]] <- 3
          zpte2 <- rep(2, nrow(dftr))
          zpte2[dfte$AHI<(AHI_boundaries[1]-Xmean[['AHI']])/Xstd[['AHI']]] <- 1
          zpte2[dfte$AHI>(AHI_boundaries[2]-Xmean[['AHI']])/Xstd[['AHI']]] <- 3
        } else {
          levels <- c(1,2,3)
          z_boundaries <- quantile(zptr, probs=c(0.25,0.75), na.rm=T)  # use boundaries based on training set
          zptr2 <- rep(2, nrow(dftr))
          zptr2[zptr<z_boundaries[1]] <- 1
          zptr2[zptr>z_boundaries[2]] <- 3
          zpte2 <- rep(2, nrow(dfte))
          zpte2[zpte<z_boundaries[1]] <- 1
          zpte2[zpte>z_boundaries[2]] <- 3
        }
        
        dfte$zp_factor <- factor(zpte2, levels)
        dfte$actual.sex <- factor(actual.sex.te, sex_levels)
        #cox_fit <- coxph(Surv(time,event)~zp_factor, data=dfte, id=id, ties='breslow')
        #cox_res <- survfit(cox_fit, data.frame(zp_factor=factor(levels, levels=levels)))
        #cox_res <- summary(cox_res)
        
        dftr$zp_factor <- factor(zptr2, levels)
        dftr$actual.sex <- factor(actual.sex.tr, sex_levels)
        cox_fit <- coxph(Surv(time,event)~zp_factor + actual.sex, data=dftr, id=id, ties='breslow')
        cox_res <- survfit(cox_fit, data.frame(
          zp_factor=factor(c(levels,levels), levels=levels),
          actual.sex=factor(c(0,0,0,1,1,1), levels=sex_levels)))
        cox_res <- summary(cox_res)
        
        #rand.idx<-sort(sample(c(1:nrow(dfte)), size=round(nrow(dfte)/10), replace=FALSE))
        #dfte2 <- dfte[rand.idx,]
        #cox_res <- survfit.par(coxph_fit,model_type,dfte,n.core=14, Ngroup=100)
        #if (model_type=='CoxPH') {
        #  surv <- list(
        #    apply(cox_res$surv[,dfte$zp_factor==1], 1, mean),
        #    apply(cox_res$surv[,dfte$zp_factor==2], 1, mean),
        #    apply(cox_res$surv[,dfte$zp_factor==3], 1, mean)
        #  )
        #  cox_res$surv <- simplify2array(surv)
        #} else {
        #  pstate <- list(
        #    apply(cox_res$pstate[,dfte$zp_factor==1,], c(1,3), mean),
        #    apply(cox_res$pstate[,dfte$zp_factor==2,], c(1,3), mean),
        #    apply(cox_res$pstate[,dfte$zp_factor==3,], c(1,3), mean)
        #  )
        #  cox_res$pstate <- aperm(simplify2array(pstate), c(1,3,2))
        #}
        
        #example.idx <- c(
        #  which.min(abs(zpte-mean(zpte[zpte2==1]))),
        #  which.min(abs(zpte-mean(zpte[zpte2==2]))),
        #  which.min(abs(zpte-mean(zpte[zpte2==3]))))
        #cox_res1 <- summary(survfit(coxph_fit[[1]], dfte[example.idx,]))
        #cox_res2 <- summary(survfit(coxph_fit[[2]], dfte[example.idx,]))
        #cox_res3 <- summary(survfit(coxph_fit[[3]], dfte[example.idx,]))
        #cox_res <- list()
        #cox_res$time <- cox_res1$time
        #if (model_type=='CoxPH') {
        #  cox_res$surv <- (cox_res1$surv + cox_res2$surv + cox_res3$surv) / 3
        #} else {
        #  cox_res$pstate <- (cox_res1$pstate + cox_res2$pstate + cox_res3$pstate) / 3
        #  cox_res$states <- cox_res1$states
        #}
        
        cox_survtime <- cox_res$time
        if (model_type=='CoxPH') {
          cox_survprob <- 1-cox_res$surv
          cox_survstate <- c('event1')
        } else {
          cox_survprob <- cox_res$pstate
          cox_survstate <- cox_res$states
        }
        cox_curve_tes[[length(cox_curve_tes)+1]] <- list(time=cox_survtime, val=cox_survprob, state=cox_survstate)
        
        # get AJ estimates only for bti==1 because it has its own CI estimate
        if (bti==1) {
          AJ_fit <- survfit(Surv(time,event)~zp_factor+actual.sex, data=dfte, id=id)
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
          AJ_curve_tes[[length(AJ_curve_tes)+1]] <- list(time=AJ_survtime, val=AJ_survprob, state=AJ_survstate, lower=AJ_lower, upper=AJ_upper)
        }
      }
      
      # end of CV
      if (bti==1) {
        print(cindex_tes)
        #print(auc_tes)
        # average over outer loop
        AJ_curve_tes <- harmonize.curves2(AJ_curve_tes)
        if (model_type=='CoxPH') {
          AJ_curve_tes$val <- apply(AJ_curve_tes$val, c(1,3), mean, na.rm=TRUE)
          AJ_curve_tes$lower <- apply(AJ_curve_tes$lower, c(1,3), mean, na.rm=TRUE)
          AJ_curve_tes$upper <- apply(AJ_curve_tes$upper, c(1,3), mean, na.rm=TRUE)
        } else {
          AJ_curve_tes$val <- apply(AJ_curve_tes$val, c(1,3,4), mean, na.rm=TRUE)
          AJ_curve_tes$lower <- apply(AJ_curve_tes$lower, c(1,3,4), mean, na.rm=TRUE)
          AJ_curve_tes$upper <- apply(AJ_curve_tes$upper, c(1,3,4), mean, na.rm=TRUE)
        }
      }
      
      # add_CV results to bootstrapping results
      cindex_tes_bt <- c(cindex_tes_bt, mean(cindex_tes, na.rm=TRUE))
      #auc_tes_bt[[length(auc_tes_bt)+1]] <- rowMeans(simplify2array(auc_tes), na.rm=TRUE)#apply(simplify2array(auc_tes), c(1,2), mean, na.rm=TRUE)
      cox_curve_tes <- harmonize.curves2(cox_curve_tes)
      if (model_type=='CoxPH') {
        cox_curve_tes$val <- apply(cox_curve_tes$val, c(1,3), mean, na.rm=TRUE)
      } else {
        cox_curve_tes$val <- apply(cox_curve_tes$val, c(1,3,4), mean, na.rm=TRUE)
      }
      cox_curve_tes_bt[[length(cox_curve_tes_bt)+1]] <- cox_curve_tes
    },
    error = function(e){ 
      print(sprintf('Error occurred in bootstrapping iteration %d', bti))
    }
  )
}
cox_curve_tes_bt <- harmonize.curves2(cox_curve_tes_bt)
#auc_tes_bt <- simplify2array(auc_tes_bt)

## after deciding the best alpha, re-fit a final model to get coef

print('refit')

X <- as.matrix(df[,c(xnames,covariate_names)])
y <- Surv(df$time, df$event)

# log-transform
#X <- sign(X)*log1p(abs(X))
# normalize
Xmean <- apply(X, 2, mean, na.rm=TRUE)
Xmean[c(rep(F,length(xnames)), covariate.is.discrete)] <- 0
Xstd <- apply(X, 2, sd, na.rm=TRUE)
Xstd[c(rep(F,length(xnames)), covariate.is.discrete)] <- 1
X_before_impute <- ( X-t(replicate(nrow(X), Xmean)) ) / t(replicate(nrow(X), Xstd))

# impute missing value
knn.model <- sklearn.impute$KNNImputer(n_neighbors=10L)
knn.model$fit(X_before_impute)
X <- knn.model$transform(X_before_impute)
#X <- impute.knn(X_before_impute, k=10)$data
df[,c(xnames,covariate_names)] <- X

#write.csv(df[,c(covariate_names, xnames, ynames)], file.path(result_path, sprintf('Rdata_%s.csv', outcome)), row.names=FALSE)
# Path as subject ID
if (exposure_type!='AHI') {
  # get best alpha from CV and bti==1
  best_alpha <- get_mode(best_alphas)
  # fit a Cox ElasticNet model to reduce dimension 
  X <- as.matrix(df[,xnames])
  # ElasticNet is only done for the outcome
  if (model_type=='CoxPH') {
    y_ <- Surv(df$time, df$event==1)
  } else {
    y_ <- Surv(df$time, df$event=='event1')
  }
  foldid <- as.numeric(fold(as.data.frame(1:nrow(df)), Ncv)$.folds)
  coxph_cv_fit <- fit_cv_cox_model(X, y_, list(xnames_f, xnames_c, xnames_o), foldid, best_alpha, n.core=Ncv)
  coef_ <- get_cv_cox_coef(coxph_cv_fit, list(xnames_mask_f, xnames_mask_c, xnames_mask_o))
  xnames2 <- c(xnames[abs(coef_)>0], covariate_names)
} else {
  xnames2 <- c(xnames, covariate_names)
  best_alpha <- NA
}

# fit competing risk model with selected features
xnames2_mask_f <- sapply(xnames2, function(x) grepl('(_F$|_F_)',x)|(x%in%covariate_names))
xnames2_mask_c <- sapply(xnames2, function(x) grepl('(_C$|_C_)',x)|(x%in%covariate_names))
xnames2_mask_o <- sapply(xnames2, function(x) grepl('(_O$|_O_)',x)|(x%in%covariate_names))
xnames2_f <- xnames2[xnames2_mask_f]
xnames2_c <- xnames2[xnames2_mask_c]
xnames2_o <- xnames2[xnames2_mask_o]

# save coefficients
if (exposure_type!='AHI') {
  coxph_fit  <- fit_model(df, list(xnames2_f, xnames2_c, xnames2_o))
  coxph_coef <- get_coef(coxph_fit, xnames2)
} else {
  coxph_fit  <- fit_model(df, list(xnames2))
  model_summary <- summary(coxph_fit[[1]])
  coxph_coef <-cbind(model_summary$coefficients, model_summary$conf.int)
  coxph_coef <- coxph_coef[, c('coef', 'Pr(>|z|)', 'lower .95', 'upper .95')]
}

if (model_type=='CoxPH') {
  zp_all <- drop(as.matrix(df[,xnames2])%*%coxph_coef[,'coef'])
} else {
  zp_all <- drop(as.matrix(df[,xnames2])%*%coxph_coef[1:(nrow(coxph_coef)/2),'coef'])
}
write.csv(coxph_coef, file.path(result_path, sprintf('coef_%s_%s.csv', outcome, model_type)))
saveRDS(coxph_fit, file.path(result_path, sprintf('model_%s_%s.rda', outcome, model_type)))

## save results

writeMat(file.path(result_path, sprintf('results_%s_%s.mat', outcome, model_type)),
         xnames=xnames2, coef=coxph_coef,
         best_l1_ratio=best_alpha, Xtr=X_before_impute,
         Xmean=Xmean, Xstd=Xstd, Xmean_names=c(xnames,covariate_names),
         zptr=zp_all,
         cindex_te=cindex_tes_bt,
         cox_curve_tes_bt_time=cox_curve_tes_bt$time,
         cox_curve_tes_bt_val=cox_curve_tes_bt$val,
         cox_curve_tes_bt_states=cox_curve_tes_bt$state,
         AJ_curve_tes_time=AJ_curve_tes$time,
         AJ_curve_tes_val=AJ_curve_tes$val,
         AJ_curve_tes_states=AJ_curve_tes$state,
         AJ_curve_tes_upper=AJ_curve_tes$upper,
         AJ_curve_tes_lower=AJ_curve_tes$lower)
#auc_te=auc_tes_bt,
