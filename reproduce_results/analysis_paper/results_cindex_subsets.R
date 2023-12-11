library(R.matlab)
library(openxlsx)
library(survival)

get.cindex <- function(model_type, df, xnames, coef) {
  if (model_type=='CoxPH_CompetingRisk') {
    zp1 <- drop(as.matrix(df[,xnames])%*%coef[1:(nrow(coef)/2),'coef'])
    zp2 <- drop(as.matrix(df[,xnames])%*%coef[(nrow(coef)/2+1):nrow(coef),'coef'])
    names(zp1) <- NULL
    names(zp2) <- NULL
    
    # reproduce coxph.fit$concordance
    #cindex <- get_concordance(df[ynames], zp)
    #cindex <- concordancefit(Surv(df$time,df$event),zp,reverse=TRUE)$concordance
    y2 <- Surv(df$time, df$event)
    y2 <- aeqSurv(y2)
    y2 <- Surv(c(y2[,1], y2[,1]),c(as.integer(y2[,2]==1), as.integer(y2[,2]==2)))
    zp <- c(zp1, zp2)
    istrat <- c(rep(1, length(zp1)), rep(2,length(zp2)))
    res <- concordancefit(y2, zp, istrat, reverse=TRUE, timefix=FALSE)
    #cindex <- c("C"=res$concordance, "se(C)"=sqrt(res$var))
  } else {
    zp1 <- drop(as.matrix(df[,xnames])%*%coef[,'coef'])
    names(zp1) <- NULL
    zp2 <- c()
    
    # reproduce coxph.fit$concordance
    y2 <- Surv(df$time, df$event)
    y2 <- aeqSurv(y2)
    res <- concordancefit(y2, zp1, reverse=TRUE, timefix=FALSE)
    #cindex_te <- c("C"=res$concordance, "se(C)"=sqrt(res$var))
  }
  return(res)
}

outcomes <- c(
  'IntracranialHemorrhage', 'IschemicStroke', 'Dementia', 'MCI+Dementia',
  'Atrial_Fibrillation', 'Myocardial_Infarction', 'DiabetesII', 'Hypertension',
  'Bipolar_Disorder', 'Depression',
  'Death'
)
exposure_type <- 'NREM'
Nbt <- 1000
base_folder <- '/data/Dropbox (BIDMC Dropbox Team)/Haoqi/SleepBasedOutcomePrediction/code-haoqi'
result_path <- file.path(base_folder, sprintf('survival_results_%s_bt%d', exposure_type, Nbt))

df.sleep.disorders <- read.csv('other_sleep_disorders.csv')
disorders <- names(df.sleep.disorders)[3:ncol(df.sleep.disorders)]
ahis <- c(0,5,10,15,20,25,30,40,50,Inf)

df.cindex <- data.frame()
df.lb <- data.frame()
df.ub <- data.frame()
df.N <- data.frame()
for (oi in 1:length(outcomes)) {
  outcome <- outcomes[oi]
  print(outcome)
  if (outcome=='Death') {
    model_type <- 'CoxPH'
  } else {
    model_type <- 'CoxPH_CompetingRisk'
  }
  mat <- readMat(file.path(result_path, sprintf('results_%s_%s.mat', outcome, model_type)))
  xnames <- unlist(mat$xnames)
  coef <- as.data.frame(mat$coef)
  names(coef) <- c('coef', 'Pr(>|z|)', 'lower .95', 'upper .95')
  Xmean <- mat$Xmean
  Xstd <- mat$Xstd
  cindex.te <- mat$cindex.te[1]
  
  df <- read.csv(file.path(result_path, sprintf('df_%s_%s.csv', outcome, model_type)))
  if (outcome!='Death') {
    df$event[df$event=='censor'] <- 0
    df$event[df$event=='event1'] <- 1
    df$event[df$event=='event2'] <- 2
    df$event <- as.integer(df$event)
    df$event <- factor(df$event, 0:2, labels=c("censor", "event1", "event2"))
  }
  delta.cindex <- cindex.te-get.cindex(model_type, df, xnames, coef)$concordance
  
  for (i in 1:(length(ahis)-1)) {
    ids <- which((df$AHI>=ahis[i])&(df$AHI<ahis[i+1]))
    res <- get.cindex(model_type, df[ids,], xnames, coef)
    val <- res$concordance+delta.cindex
    err <- sqrt(res$cvar)*1.96
    df.cindex[oi, sprintf('%g<=AHI<%g', ahis[i], ahis[i+1])] <- val
    df.lb[oi, sprintf('%g<=AHI<%g', ahis[i], ahis[i+1])] <- max(0,val-err)
    df.ub[oi, sprintf('%g<=AHI<%g', ahis[i], ahis[i+1])] <- min(1,val+err)
    df.N[oi, sprintf('%g<=AHI<%g', ahis[i], ahis[i+1])] <- length(ids)
  }
  
  df2 <- merge(df, df.sleep.disorders, by='MRN', all.x=T)
  for (disorder in disorders) {
    ids <- which(df2[,disorder]==1)
    res <- get.cindex(model_type, df[ids,], xnames, coef)
    val <- res$concordance+delta.cindex
    err <- sqrt(res$cvar)*1.96
    df.cindex[oi, disorder] <- val
    df.lb[oi, disorder] <- max(0,val-err)
    df.ub[oi, disorder] <- min(1,val+err)
    df.N[oi, disorder] <- length(ids)
  }
}
df.cindex <- cbind(outcomes, df.cindex)
df.lb <- cbind(outcomes, df.lb)
df.ub <- cbind(outcomes, df.ub)
df.N <- cbind(outcomes, df.N)
write.xlsx(list('cindex'=df.cindex, 'lb'=df.lb, 'ub'=df.ub, 'N'=df.N), file='cindex_subsets.xlsx')
