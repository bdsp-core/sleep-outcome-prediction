library(reticulate)

get_mode <- function(v){
  uniqv <- unique(v)
  res <- uniqv[which.max(tabulate(match(v, uniqv)))]
  return(res)
}


# fit separate models using same features from separate channels
fit_cv_cox_model <- function(X, y, xnames, foldid, alpha, n.core=1) {
  cl <- makeCluster(n.core)
  registerDoParallel(cl)
  
  res <- list()
  for (i in 1:length(xnames)) {
    coxph_fit <- cv.glmnet(X[,xnames[[i]]], y, family='cox', type.measure='C', foldid=foldid, alpha=alpha, parallel=TRUE)
    res[[length(res)+1]] <- coxph_fit
  }
  
  stopCluster(cl)
  return(res)
}


get_cvm <- function(model) {
  cvms <- c()
  for (i in 1:length(model))
    cvms <- c(cvms, model[[i]]$cvm[model[[i]]$index['min',]])
  cvm <- mean(cvms)
  return(cvm)
}


get_cv_cox_coef <- function(model, xnames_masks) {
  coefs <- rep(0, length(xnames_masks[[1]]))
  for (i in 1:length(model))
    coefs[xnames_masks[[i]]] <- as.numeric(coef(model[[i]], s=model[[i]]$lambda[model[[i]]$index['min',]]))
  coefs <- coefs/length(model)
  return(coefs)
}


fit_competing_risk_model <- function(df, xnames) {
  res <- list()
  for (i in 1:length(xnames)) {
    formula_ <- as.formula(paste('Surv(time, event)~', paste(xnames[[i]], collapse = "+")))
    res[[length(res)+1]] <- coxph(formula_, data=df, id=id, ties='breslow')
  }
  
  return(res)
}


get_competing_risk_coef <- function(model, xnames) {
  coefs <- c()
  for (i in 1:length(model)) {
    model_summary <- summary(model[[i]])
    coefs <- rbind(coefs, cbind(model_summary$coefficients, model_summary$conf.int))
  }
  
  coefs <- rbind(coefs[row.names(coefs)!='Age_1:2',], colMeans(coefs[row.names(coefs)=='Age_1:2',,drop=FALSE]))
  coefs <- rbind(coefs[row.names(coefs)!='Sex_1:2',], colMeans(coefs[row.names(coefs)=='Sex_1:2',,drop=FALSE]))
  coefs <- rbind(coefs[row.names(coefs)!='Age_1:3',], colMeans(coefs[row.names(coefs)=='Age_1:3',,drop=FALSE]))
  coefs <- rbind(coefs[row.names(coefs)!='Sex_1:3',], colMeans(coefs[row.names(coefs)=='Sex_1:3',,drop=FALSE]))
  rnames <- row.names(coefs)
  rnames[(length(rnames)-3):length(rnames)] <- c('Age_1:2', 'Sex_1:2', 'Age_1:3', 'Sex_1:3')
  row.names(coefs) <- rnames
  
  rnames <- c(sapply(xnames, function(x) paste(x, '_1:2',sep='')), sapply(xnames, function(x) paste(x, '_1:3',sep='')))
  coefs <- coefs[rnames,]
  
  coefs <- coefs[, c('coef', 'Pr(>|z|)', 'lower .95', 'upper .95')]
  coefs[,'coef'] <- coefs[,'coef']/length(model)
  coefs[,'lower .95'] <- exp(log(coefs[,'lower .95'])/length(model))
  coefs[,'upper .95'] <- exp(log(coefs[,'upper .95'])/length(model))
  return(coefs)
}


fit_cox_model <- function(df, xnames) {
  res <- list()
  for (i in 1:length(xnames)) {
    formula_ <- as.formula(paste('Surv(time, event)~', paste(xnames[[i]], collapse = "+")))
    res[[length(res)+1]] <- coxph(formula_, data=df, ties='breslow')
  }
  return(res)
}


get_cox_coef <- function(model, xnames) {
  coefs <- c()
  for (i in 1:length(model)) {
    model_summary <- summary(model[[i]])
    coefs <- rbind(coefs, cbind(model_summary$coefficients, model_summary$conf.int))
  }
  
  coefs <- rbind(coefs[row.names(coefs)!='Age',], colMeans(coefs[row.names(coefs)=='Age',,drop=FALSE]))
  coefs <- rbind(coefs[row.names(coefs)!='Sex',], colMeans(coefs[row.names(coefs)=='Sex',,drop=FALSE]))
  rnames <- row.names(coefs)
  rnames[(length(rnames)-1):length(rnames)] <- c('Age', 'Sex')
  row.names(coefs) <- rnames
  coefs <- coefs[xnames,]
  
  coefs <- coefs[, c('coef', 'Pr(>|z|)', 'lower .95', 'upper .95')]
  coefs[,'coef'] <- coefs[,'coef']/length(model)
  coefs[,'lower .95'] <- exp(log(coefs[,'lower .95'])/length(model))
  coefs[,'upper .95'] <- exp(log(coefs[,'upper .95'])/length(model))
  return(coefs)
}


add_noise_event_time <- function(df) {
  ids <- (df$event1_occur==0)
  df[ids,'event1_time'] <- pmax(0, df[ids,'event1_time'] + rnorm(sum(ids))*1/365)
  df[ids,'event2_time'] <- df[ids,'event1_time']
  
  ids <- (df$event1_occur==1)
  dt <-  df[ids,'event2_time'] - df[ids,'event1_time']
  dt <- pmax(0, dt + rnorm(sum(ids))*1/365)
  df[ids,'event1_time'] <- pmax(0, df[ids,'event1_time'] + rnorm(sum(ids))*1/365)
  df[ids,'event2_time'] <- df[ids,'event1_time'] + dt
  return(df)
}


harmonize.curves <- function(times, vals, pre_fill_val=0) {
  # times = list( (#T1,), (#T2,), ... (#T(level),))
  # vals  = list( (#T1,K), (#T2,K), ... (#T(level), K))
  # convert into:
  # time  = (#T,)
  # val   = (#T, #level, K)
  # if K=1, val is squeezed to (#T, #level)
  Nlevel <- length(times)
  stopifnot(Nlevel==length(vals))
  
  # find all unique times
  all_times <- sort(Reduce(union, times))
  T <- length(all_times)
  dimension <- dim(as.array(vals[[1]]))
  Ndim <- length(dimension)
  if (Ndim==1) {
    K <- 1
  } else if (Ndim==2) {
    K <- dimension[2]
  } else {
    origin.shape <- dimension[2:Ndim]
    K <- prod(origin.shape)
  }
  all_vals <- array(NA, c(T, Nlevel, K))
  
  for (i in 1:Nlevel) {
    idx <- findInterval(times[[i]], all_times)
    if (length(idx)==0) {
      all_vals[,i,] <- NA
    } else {
      if (Ndim<=2) {
        val <- as.matrix(vals[[i]])
      } else {
        val <- array_reshape(as.array(vals[[i]]), c(dim(vals[[i]])[1], K))
      }
      val[is.na(val)] <- -999  # convert NA to -999 since stepfun does not handle NA
      for (j in 1:K) {
        fn <- stepfun(idx, c(pre_fill_val, val[,j]))
        fn_val <- fn(1:T)
        fn_val[fn_val==-999] <- NA
        all_vals[,i,j] <- fn_val
      }
    }
  }
  
  if (Ndim==1) {
    all_vals <- all_vals[,,1]
  } else if (Ndim>2) {
    new_shape <- c(c(T, Nlevel), origin.shape)
    all_vals <- array_reshape(all_vals, new_shape)
  }
  res <- list()
  res$time <- all_times
  res$val <- all_vals
  return(res)
}


harmonize.curves2 <- function(curves, pre_fill_val=0) {
  # converts list( list(time, val, state, [lower, upper]), list(time, val, state, [lower, upper]), ... )
  # into the input format of harmonize.curves:
  # time = list( time, time, ... )
  # val = list( time, time, ... ) for val and [lower, upper]
  Ncurve <- length(curves)
  times <- list()
  vals <- list()
  lowers <- list()
  uppers <- list()
  states <- list()
  
  for (i in 1:Ncurve) {
    states[[length(states)+1]] <- curves[[i]]$state
    times[[length(times)+1]] <- curves[[i]]$time
    vals[[length(vals)+1]] <- curves[[i]]$val
    if ('lower' %in% names(curves[[i]])) {
      lowers[[length(lowers)+1]] <- curves[[i]]$lower
    }
    if ('upper' %in% names(curves[[i]])) {
      uppers[[length(uppers)+1]] <- curves[[i]]$upper
    }
  }
  
  res <- harmonize.curves(times, vals, pre_fill_val=pre_fill_val)
  if (length(lowers)==length(curves)) {
    res_ <- harmonize.curves(times, lowers, pre_fill_val=pre_fill_val)
    stopifnot(all(res$time==res_$time))
    res$lower <- res_$val
  }
  if (length(uppers)==length(curves)) {
    res_ <- harmonize.curves(times, uppers, pre_fill_val=pre_fill_val)
    stopifnot(all(res$time==res_$time))
    res$upper <- res_$val
  }
  
  res$states <- curves[[1]]$state #TODO check len(set(states))==1
  return(res)
}

harmonize.curves.AJ <- function(AJ_res, levels) {
  # convert dim(AJ_res$pstate) = (#T, K) --> 
  # list( (#T1,K), (#T2,K), ... (#T(level), K)) -->
  # (#T, #level, K)
  times <- list()
  vals <- list()
  lowers <- list()
  uppers <- list()
  for (level in levels) {
    strata_ids <- AJ_res$strata==level
    times[[length(times)+1]] <- AJ_res$time[strata_ids]
    if (attributes(AJ_res)$class=="summary.survfitms") {
      vals[[length(vals)+1]] <- AJ_res$pstate[strata_ids,,drop=FALSE] # prevents when(strata_ids)==1, becomes a vector
      if ('lower' %in% attributes(AJ_res)$names) {
        lowers[[length(lowers)+1]] <- AJ_res$lower[strata_ids,,drop=FALSE]
      }
      if ('upper' %in% attributes(AJ_res)$names) {
        uppers[[length(uppers)+1]] <- AJ_res$upper[strata_ids,,drop=FALSE]
      }
    } else{
      vals[[length(vals)+1]] <- AJ_res$surv[strata_ids]
      if ('lower' %in% attributes(AJ_res)$names) {
        lowers[[length(lowers)+1]] <- AJ_res$lower[strata_ids]
      }
      if ('upper' %in% attributes(AJ_res)$names) {
        uppers[[length(uppers)+1]] <- AJ_res$upper[strata_ids]
      }
    }
  }
  
  if (attributes(AJ_res)$class=="summary.survfitms") {
    pre_fill_val <- 0
  } else {
    pre_fill_val <- 1
  }
  res_ <- harmonize.curves(times, vals, pre_fill_val=pre_fill_val)
  res <- list()
  res$time <- res_$time
  if (attributes(AJ_res)$class=="summary.survfitms") {
    res$pstate <- res_$val
  } else {
    res$surv <- res_$val
  }
  if (length(lowers)==length(times)) {
    res$lower <- harmonize.curves(times, lowers, pre_fill_val=pre_fill_val)$val
  }
  if (length(uppers)==length(times)) {
    res$upper <- harmonize.curves(times, uppers, pre_fill_val=pre_fill_val)$val
  }

  if (attributes(AJ_res)$class=="summary.survfitms") {
    res$states <- AJ_res$states
  } else {
    res$states <- c('event1')
  }
  
  return(res)
}


survfit.par <- function(model, model.type, df, n.core=8, Ngroup=8) {
  cl <- makeCluster(n.core)
  registerDoParallel(cl)
  
  groups <- group(1:nrow(df),Ngroup)
  
  res <- foreach(i=1:Ngroup, .packages='survival') %dopar% {
    res_ <- 0
    for (ii in 1:length(model)) {
      tmp <- summary(survfit(model[[ii]], df[groups$.groups==i,]))
      if (model.type=='CoxPH') {
        res_ <- res_ + tmp$surv
      } else {
        res_ <- res_ + tmp$pstate
      }
    }
    res_ <- res_/length(model)
    res_ 
  }
  stopCluster(cl)
  
  res1 <- survfit(model[[1]], df[1,])
  res1 <- summary(res1)
  
  result <- list()
  result$time <- res1$time
  if (model.type=='CoxPH') {
    result$surv <- abind(res,along=2)
  } else {
    result$pstate <- abind(res,along=2)
    result$states <- res1$states
  }
    
  return(result)
}
