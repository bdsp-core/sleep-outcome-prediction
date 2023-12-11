library(survival)
library(readxl)
library(reticulate)  # for sklearn.impute.KNNImputer
source('Rfunctions.R')


outcomes <- c(
  'IntracranialHemorrhage', 'IschemicStroke', 'Dementia', 'MCI+Dementia',
  'Atrial_Fibrillation', 'Myocardial_Infarction', 'DiabetesII', 'Hypertension',
  'Bipolar_Disorder', 'Depression',
  'Death'
)
covariate_names <- c('Age', 'Sex', 'BMI', 'MedBenzo', 'MedAntiDep', 'MedSedative', 'MedAntiEplipetic', 'MedStimulant')
covariate.is.discrete <- c(F, T, F, T, T, T, T, T)

base_folder <- '/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction (1)/code-haoqi'
result_path <- 'survival_results_NREM'
random_seed <- 2022

dfX <- read.csv(file.path(base_folder, '../shared_data/MGH/to_be_used_features_NREM.csv'))
dfX <- subset(dfX, select=-c(DateOfVisit, TwinDataID, Path) )

stopifnot(names(dfX)[12]=='TotalSleepTime')
all.Xnames <- names(dfX)[12:ncol(dfX)]

for (outcome in outcomes) {
  print(outcome)

  if (outcome=='Death') {
    dfy <- read_excel(file.path(base_folder, 'time2event_IschemicStroke.xlsx'))
  } else {
    dfy <- read_excel(file.path(base_folder, sprintf('time2event_%s.xlsx', outcome)))
  }
  stopifnot(all(dfX$MRN==dfy$MRN))
  #stopifnot(all(dfX$DateOfVisit==dfy$DateOfVisit))
  dfy <- subset(dfy, select=-c(PatientID, MRN, DateOfVisit) )
  df.all <- cbind(dfX, dfy)
  
  # generate time and event
  ynames <- c('time', 'event')
  if (outcome=='Death') {
    model_type <- 'CoxPH'
    # only look at future events relative to sleep study
    ids <- df.all$time_death>0
    df.all <- df.all[ids,]
    
    names(df.all)[names(df.all)=="time_death"] <- "time"
    names(df.all)[names(df.all)=="cens_death"] <- "event"
    df.all$event <- 1-df.all$event
    df.all <- subset(df.all, select = -c(time_outcome, cens_outcome) )
  } else {
    model_type <- 'CoxPH_CompetingRisk'
    # only look at future events relative to sleep study
    ids <- (!is.na(df.all$cens_death))&(df.all$time_death>0)&(!is.na(df.all$cens_outcome))&(df.all$time_outcome>0)
    df.all <- df.all[ids,]
    
    df.all$event2_occur = 1-df.all$cens_death
    df.all$event1_occur = 1-df.all$cens_outcome
    df.all$time <- with(df.all, ifelse(event1_occur==0, time_death, time_outcome))
    event <- with(df.all, ifelse(event1_occur==0, 2*event2_occur, 1))
    df.all$event <- factor(event, 0:2, labels=c("censor", "event1", "event2"))
    df.all <- subset(df.all, select = -c(time_death, time_outcome, cens_death, cens_outcome, event1_occur, event2_occur) )
  }
  df.all$id <- 1:nrow(df.all)
  rownames(df.all) <- NULL
  if (model_type=='CoxPH') {
    fit_model <- fit_cox_model
    get_coef  <- get_cox_coef
  } else {
    fit_model <- fit_competing_risk_model
    get_coef  <- get_competing_risk_coef
  }

  df.coef <- read.csv(file.path(base_folder, sprintf('survival_results_NREM/coef_%s_%s.csv', outcome, model_type)))
  tostudy.Xnames <- df.coef$X
  if (outcome!='Death') {
    tostudy.Xnames <- tostudy.Xnames[grep('_1:2', tostudy.Xnames)]
    tostudy.Xnames <- sub('_1:2', '', tostudy.Xnames)
  }
  tostudy.Xnames <- setdiff(tostudy.Xnames, covariate_names)
  
  coefs.tostudy <- c()
  for (col in tostudy.Xnames) {
    print(col)
    df <- df.all[!is.na(df.all[,col]),]
    Xnames <- c(col, covariate_names)
    
    X <- as.matrix(df[,Xnames])
    y <- Surv(df$time, df$event)
    
    nonconstant_cols <- apply(X, 2, sd, na.rm=TRUE)>0
    X <- X[, nonconstant_cols]
    Xnames <- Xnames[nonconstant_cols]
    
    # log-transform
    #X <- sign(X)*log1p(abs(X))
    # normalize
    Xmean <- apply(X, 2, mean, na.rm=TRUE)
    Xmean[c(rep(F,length(col)), covariate.is.discrete)] <- 0
    Xstd <- apply(X, 2, sd, na.rm=TRUE)
    Xstd[c(rep(F,length(col)), covariate.is.discrete)] <- 1
    X_before_impute <- ( X-t(replicate(nrow(X), Xmean)) ) / t(replicate(nrow(X), Xstd))
    
    # impute missing value
    sklearn.impute <- import("sklearn.impute")
    set.seed(random_seed)
    knn.model <- sklearn.impute$KNNImputer(n_neighbors=10L)
    knn.model$fit(X_before_impute)
    X <- knn.model$transform(X_before_impute)
    #X <- impute.knn(X_before_impute, k=10)$data
    df[,Xnames] <- X
    
    # fit competing risk model with selected features
    
    coxph_fit  <- fit_model(df, list(Xnames))
    coxph_coef <- get_coef(coxph_fit, Xnames)
    coxph_coef <- cbind(coxph_coef, N=nrow(df))
    if (outcome!='Death') {
      col2 <- paste(col,'_1:2',sep='')
    } else {
      col2 <- col
    }
    coefs.tostudy <- rbind(coefs.tostudy, coxph_coef[col2,])
  }
  
  coefs.tostudy <- cbind(data.frame(tostudy.Xnames),coefs.tostudy)
  coefs.tostudy <- merge(x=data.frame(tostudy.Xnames=all.Xnames), y=coefs.tostudy, by="tostudy.Xnames", all.x=TRUE)
  names(coefs.tostudy)[names(coefs.tostudy) == 'tostudy.Xnames'] <- 'Xname'
  print(coefs.tostudy)
  filename <- sprintf('univariate_coefs_%s.csv', outcome)
  write.csv(coefs.tostudy, file.path(base_folder, result_path, filename), row.names=FALSE, na='')
}
