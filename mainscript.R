###
# DATA MINING - THESIS PROJECT  ####
# Version 2
# July 2019
# By Robby De Pauw                                                             
###

## PREPARATION (INSTALLING REQUIRED PACKAGES) ####

.libPaths("C:/Users/rdpauw/R_library")

#install.packages('tidyverse', lib = "C:/Users/rdpauw/R_library")
library(tidyverse)
#install.packages('DMwR', lib = "C:/Users/rdpauw/R_library")
library(DMwR)
#install.packages('ipred', lib = "C:/Users/rdpauw/R_library")
library(ipred)
#install.packages('caret', lib = "C:/Users/rdpauw/R_library")
library(caret)
#install.packages('mlbench', lib = "C:/Users/rdpauw/R_library")
library(mlbench)
#install.packages('ROSE', lib = "C:/Users/rdpauw/R_library")
library(ROSE)
#install.packages("ggplot2", lib = "C:/Users/rdpauw/R_library")
library(ggplot2)
#install.packages("xtable", lib = "C:/Users/rdpauw/R_library")
library(xtable)
#install.packages("vcd", lib = "C:/Users/rdpauw/R_library")
library(vcd)
#install.packages("VIM", lib = "C:/Users/rdpauw/R_library")
library(VIM)
#install.packages("corrplot", lib = "C:/Users/rdpauw/R_library")
library(corrplot)
library(glmnet)
#install.packages("C50", lib = "C:/Users/rdpauw/R_library")
library(C50)
#install.packages("stabm", lib = "C:/Users/rdpauw/R_library")
library(stabm)
#install.packages("BiocManager")
library(BiocManager)
#install("OmicsMarkeR")
library(OmicsMarkeR)

###
# PART I: CLEANING THE DATA ####
###

# SELECT DATABASE
file_id <- file.choose() # SELECT DATABASE ON ALL COMPUTERS
#setwd('c://Users/rdpauw/OneDrive - UGent/MSc in Statistical Data Analysis/Thesis/Nekpijn/')
#setwd('/Users/robbydepauw/OneDrive - UGent/MSc in Statistical Data Analysis/Thesis/Nekpijn/')

# Source scripts
source('getStability.R')
source('hypothesisTestCompareStabilities.R')
source('summaryresults.R')
source('listfeats.R')
source('calculate_cramer.R')

# READ-IN DATABASE
pain <- read.csv2(file_id, na.strings = c(" ", "NA"))
#pain <- read.csv2('database.csv', na.strings = c(" ", "NA"))

# DISCARD INCOMPLETE OBSERVATIONS
id <- which(complete.cases(pain))
length(id) # 163 full cases with no missing data

missing_pain <- pain[-id,]
pain <- pain[id,]

# CHECK DATASET STRUCTURE
str(pain)

# Outcome variables in tables
table(pain$DOM.NOCI)

# Take nociceptive as outcome
outcome <- pain$DOM.NOCI

# remove other outcome variables [1:6]
pain <- pain[,-c(1:6)]
str(pain) # Evaluate the structure of the included variables

# Create factors (first 7 vars are continuous)
id <- 8:dim(pain)[2]
pain[,id] <- lapply(pain[,id], as.factor)
# Others should be numerical
pain[,-id] <- lapply(pain[,-id], as.numeric)
str(pain)

# PART II: DESCRIPTIVES ####
table(outcome)
prop.table(table(outcome)) # Descritives of the outcome measure

tab_demo_1 <- pain %>%
  summarise_if(is.numeric, list(~mean(.), ~sd(.), ~min(.), ~max(.)), na.rm = TRUE) %>%
  gather() %>%
  separate(key, c("var","type"), sep = "_") %>%
  spread(type, value) # Descriptives of the continuous variables

xtable(tab_demo_1)

tab_demo_2 <- summary(pain)[,id] # Descriptives of categorical variables
xtable(t(tab_demo_2))

# DROP constant factors
nzv <- nearZeroVar(pain, saveMetrics= TRUE) # Find variables that are constant
nzv[nzv$nzv,]
xtable(nzv[nzv$nzv,]) # Create table for Latex
nearzero.id <- which(nzv$nzv == TRUE)
painfilter <- pain[, -nearzero.id] # Delete these variables from SET
# add the outcome
painfilter <- cbind(outcome, painfilter) # Combine outcome with new set
colnames(painfilter)

# FIND CORRELATED PREDICTORS ####
id <- sapply(painfilter, is.numeric)
id <- as.numeric(which(id == TRUE))
findCorrelation(cor(painfilter[,id]), cutoff = 0.9) # correlations higher than 0.9 are to be discarded before analysis by the models
df <- painfilter[,-id]
# Initialize empty matrix to store coefficients
empty_m <- matrix(ncol = length(df),
                  nrow = length(df),
                  dimnames = list(names(df), 
                                  names(df)))

cor_matrix <- calculate_cramer(empty_m ,df)
id <- lower.tri(cor_matrix, diag = FALSE) # Only evaluate half the matrix
cor.id <- which(cor_matrix[id] > 0.9)

## SCALE and CENTER continuous variables
painfilter[,id] <- sapply(painfilter[,id], scale)

# Write away the result
#write.csv2(painfilter, 'pain_cleaned.csv')

# PART III: Model Buiding ####
# setwd('c://Users/rdpauw/OneDrive - UGent/MSc in Statistical Data Analysis/Thesis/')
# pain <- read.csv2('pain_cleaned.csv')
# pain <- pain[,-1]
# pain[,-1] <- lapply(pain[-1], as.numeric)

# SPLIT THE DATASET ####
# Split data set into training and test set
set.seed(666)
id <- createDataPartition(painfilter$outcome, p = 1/3, list = FALSE)

pain.train <- painfilter[id,]
pain.test <- painfilter[-id,]
rm(id)

# Data balance within train and test dataset
prop.table(table(painfilter$outcome))
prop.table(table(pain.train$outcome))
prop.table(table(pain.test$outcome))

# Number of datapoints in the train and test dataset
n.train <- nrow(pain.train)
n.test <- nrow(pain.test)

# BALANCE THE DATASET ####

# Check the performance
trainX <- pain.train[,-1]
trainY <- pain.train[,1]

testX <- pain.test[,-1]
testY <- pain.test[,1]

## LOOP OVER DIFFERENT MODELS
for (s in c("none","rose","smote")) {
  
  print(s)
  
  # Get the correct settings
  
  if (s == "none") {
    cctrl1 <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 100, returnResamp="all",
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)
  } else {
    cctrl1 <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 100, returnResamp="all",
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary,
                           sampling = s)    
  }

  ## GLMNET
  model_glmnet <- train(x = data.matrix(trainX), y = trainY, 
                        method = "glmnet", 
                        trControl = cctrl1,
                        metric = "ROC")
  
  assign(x = paste0("model_glmnet_",s), model_glmnet)
  
  ## RANDOM FOREST
  model_rf <- train(data.matrix(trainX), trainY, method = "rf", 
                    trControl = cctrl1, metric = "ROC", 
                    ntree = 500, nodesize = 1)
  
  assign(x = paste0("model_rf_",s), model_rf)
  
  ## SPARSE LDA
  
  model_sparseLDA <- train(data.matrix(trainX), trainY, method = "sparseLDA", 
                           trControl = cctrl1,metric = "ROC")
  
  assign(x = paste0("model_sparseLDA_",s), model_sparseLDA)
  
  ## C5.0Tree
  model_C5.0Tree<- train(data.matrix(trainX), trainY, method = "C5.0Tree", 
                         trControl = cctrl1,metric = "ROC")

  assign(x = paste0("model_C5.0Tree_",s), model_C5.0Tree)
    
  print(paste(s,"is finished!"))
  
  }

## COMPARISON of models based on CROSS VALIDITATION ####

## none
resamps <- resamples(list(GLMNET = model_glmnet_none,
                          RF = model_rf_none,
                          sLDA = model_sparseLDA_none,
                          C50 = model_C5.0Tree_none))

summary(resamps)

## Creating a plot
theme1 <- trellis.par.get() #Settings plot parameters
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
bwplot(resamps, layout = c(3, 1))

## ROSE
resamps <- resamples(list(GLMNET = model_glmnet_rose,
                          RF = model_rf_rose,
                          sLDA = model_sparseLDA_rose,
                          C50 = model_C5.0Tree_rose))

resamps
summary(resamps)

bwplot(resamps, layout = c(3, 1))

## SMOTE
resamps <- resamples(list(GLMNET = model_glmnet_smote,
                          RF = model_rf_smote,
                          sLDA = model_sparseLDA_smote,
                          C50 = model_C5.0Tree_smote))

resamps
summary(resamps)

bwplot(resamps, layout = c(3, 1))

## RF
resamps <- resamples(list(NONE = model_rf_none,
                          ROSE = model_rf_rose,
                          SMOTE = model_rf_smote))

resamps
summary(resamps)

## SIMULATION by Bootstrapping####
## ASSESS THE ACCURACY AND STABILITY OF FEATURE SETS ACROSS DIFFERENT SAMPLING TECHNIQUES

# Models to evaluate
sim_models <- c("glmnet", "rf", "sparseLDA", "C5.0Tree")

## Check accuracy over different samples
nsim <- 100

for (m in sim_models) {
  
    res <- data.frame(type = NA, accuracy = NA, sens = NA, spec = NA, 
                      tp = NA, tn = NA, fp = NA, fn = NA)
    n_features <- data.frame(type = NA, n = NA)
    
    imp <- matrix(NA, ncol = 10, nrow = nsim)
    imp_smote <- matrix(NA, ncol = 10, nrow = nsim)
    imp_rose <- matrix(NA, ncol = 10, nrow = nsim)

      for (i in 1:nsim) {
        set.seed(i+102)
        temp <- data.frame(type = NA, accuracy = NA, sens = NA, spec = NA, 
                           tp = NA, tn = NA, fp = NA, fn = NA)
        n_temp <- data.frame(type = NA, n = NA)
      
        # Resample training dataset
        id <- createDataPartition(painfilter$outcome, p = 3/4, list = FALSE)
        pain.train <- painfilter[id,]
        pain.test <- painfilter[-id,]

        # Specify CV
        cctrl <- trainControl(method="cv", number=5, returnResamp="all")
        
        # Normal
        X <- pain.train[,-1]
        Y <- pain.train$outcome
        
        ## Build the model
        model <- train(data.matrix(X), Y, method = m, 
                              trControl = cctrl)
        ## Calculate performance of the model
        tmp_conf <- confusionMatrix(pain.test$outcome,
                                    predict.train(model, 
                                                  newdata = data.matrix(pain.test[,-1]),
                                                                    type = "raw"))
        ## Assess variable importance (top 10)
        names_imp <- row.names(varImp(model)$importance) # Store the parameter-names
        imp_zero_id <- which(varImp(model)$importance == 0) ## Exclude parameters not in model
        nonzero_imp <- varImp(model)$importance[-imp_zero_id,]
        nonzero_names <- names_imp[-imp_zero_id]
        
        tmp_imp <-  nonzero_names[order(nonzero_imp, decreasing = TRUE)][1:10]
        
        ## SAVE the number of features in final model
        n_temp[1,] <- c("normal",length(nonzero_names)) 
        
        ## SAVE results
        temp[1,] <- c("normal", tmp_conf$overall[1], tmp_conf$byClass[1:2],
                      tmp_conf$table[1,1], tmp_conf$table[2,2],
                      tmp_conf$table[1,2], tmp_conf$table[2,1])
        imp[i,] <- tmp_imp
        
        # SMOTE
        SMOTE_pain <- SMOTE(outcome~., data = pain.train)
        SMOTEX <- SMOTE_pain[,-1]
        SMOTEY <- SMOTE_pain[,1]
  
        ## Build the model
        model <- train(data.matrix(SMOTEX), SMOTEY, method = m, 
                              trControl = cctrl)
        ## Calculate performance of the model
        tmp_conf_smote <- confusionMatrix(pain.test$outcome,
                                          predict.train(model, 
                                                        newdata = data.matrix(pain.test[,-1]),
                                                        type = "raw"))
        ## Assess variable importance (top 10)
        names_imp <- row.names(varImp(model)$importance) # Store the parameter-names
        imp_zero_id <- which(varImp(model)$importance == 0) ## Exclude parameters not in model
        nonzero_imp <- varImp(model)$importance[-imp_zero_id,]
        nonzero_names <- names_imp[-imp_zero_id]
        
        tmp_imp_smote <-  nonzero_names[order(nonzero_imp, decreasing = TRUE)][1:10]

        ## SAVE the number of features in final model
        n_temp[2,] <- c("smote",length(nonzero_names)) 
        
        ## SAVE results
        temp[2,] <- c("smote", tmp_conf_smote$overall[1], tmp_conf_smote$byClass[1:2],
                      tmp_conf_smote$table[1,1], tmp_conf_smote$table[2,2],
                      tmp_conf_smote$table[1,2], tmp_conf_smote$table[2,1])        
        imp_smote[i,] <- tmp_imp_smote
        
        # ROSE
        ROSE_pain <- ROSE(outcome~., data = pain.train)$data
        ROSEX <- ROSE_pain[,-1]
        ROSEY <- ROSE_pain[,1]
        
        ## Build the model
        model <- train(data.matrix(ROSEX), ROSEY, method = m, 
                              trControl = cctrl)
        ## Calculate performance of the model
        tmp_conf_rose <- confusionMatrix(pain.test$outcome,
                                         predict.train(model, 
                                                       newdata = data.matrix(pain.test[,-1]),
                                                       type = "raw"))
        ## Assess variable importance (top 10)
        names_imp <- row.names(varImp(model)$importance) # Store the parameter-names
        imp_zero_id <- which(varImp(model)$importance == 0) ## Exclude parameters not in model
        nonzero_imp <- varImp(model)$importance[-imp_zero_id,]
        nonzero_names <- names_imp[-imp_zero_id]
        
        tmp_imp_rose <-  nonzero_names[order(nonzero_imp, decreasing = TRUE)][1:10]
        
        ## SAVE the number of features in final model
        n_temp[3,] <- c("rose",length(nonzero_names)) 
        
        ## SAVE results
        temp[3,] <- c("rose", tmp_conf_rose$overall[1], tmp_conf_rose$byClass[1:2],
                      tmp_conf_rose$table[1,1], tmp_conf_rose$table[2,2],
                      tmp_conf_rose$table[1,2], tmp_conf_rose$table[2,1])        
        imp_rose[i,] <- tmp_imp_rose
        
        # merge datafram
        res <- rbind(res, temp)
        n_features <- rbind(n_features, n_temp)
        
        # print iteration
        print(paste(m,i/nsim))
}

    assign(x = paste0(m,"_res"), value = res)
    assign(x = paste0(m,"_nfeatures"), value = n_features)
    
    features <- list(norm=t(imp), smote=t(imp_smote), rose = t(imp_rose))
    assign(x = paste0(m,"_features"), value = features)
    
    
}

## Script for Nogueira, Sechidis and Brown's stability measure
for (m in sim_models){
  
  tmp <- get(paste0(m,"_features"))
  nogue_matrix_norm <- matrix(NA, nrow = 100, ncol = 69)
  nogue_matrix_rose <- matrix(NA, nrow = 100, ncol = 69)
  nogue_matrix_smote <- matrix(NA, nrow = 100, ncol = 69)
  
  for (i in 1:100) {
    id_1 <- which(colnames(trainX) %in% tmp$norm[,i])
    nogue_matrix_norm[i,id_1] <- 1 
    nogue_matrix_norm[i,-id_1] <- 0
  }
  
  for (i in 1:100) {
    id_1 <- which(colnames(trainX) %in% tmp$rose[,i])
    nogue_matrix_rose[i,id_1] <- 1 
    nogue_matrix_rose[i,-id_1] <- 0
  }

  for (i in 1:100) {
    id_1 <- which(colnames(trainX) %in% tmp$smote[,i])
    nogue_matrix_smote[i,id_1] <- 1 
    nogue_matrix_smote[i,-id_1] <- 0
  }  
  
  assign(x = paste0(m,"_nogue_norm"), value = nogue_matrix_norm)
  assign(x = paste0(m,"_nogue_rose"), value = nogue_matrix_rose)
  assign(x = paste0(m,"_nogue_smote"), value = nogue_matrix_smote)
}

## RESULTS of BOOTRSTRAPPING ALGORITHM ####

## PERFORMANCE

#GLMNET
glmnet_res[,-1] <- sapply(glmnet_res[,-1], function(x) as.numeric(as.character(x)))
glmnet_res$AUC <- (glmnet_res$sens + glmnet_res$spec)/2
glmnet_res$type <- fct_explicit_na(glmnet_res$type)

summaryresults(glmnet_res, "glmnet.csv") # Create a summary of the results

glmnet.plot <- gather(glmnet_res[,-c(2,5:8)], "meas", "value", 2:4) # Create a dataframe for later plotting

#RF
rf_res[,-1] <- sapply(rf_res[,-1], function(x) as.numeric(as.character(x)))
rf_res$AUC <- (rf_res$sens + rf_res$spec)/2
rf_res$type <- fct_explicit_na(rf_res$type)

summaryresults(rf_res, "rf.csv")

rf.plot <- gather(rf_res[,-c(2,5:8)], "meas", "value", 2:4)

#sLDA
sparseLDA_res[,-1] <- sapply(sparseLDA_res[,-1], function(x) as.numeric(as.character(x)))
sparseLDA_res$AUC <- (sparseLDA_res$sens + sparseLDA_res$spec)/2
sparseLDA_res$type <- fct_explicit_na(sparseLDA_res$type)

summaryresults(sparseLDA_res, "sLDA.csv")

sparseLDA.plot <- gather(sparseLDA_res[,-c(2,5:8)], "meas", "value", 2:4)

#C5.0
C5.0Tree_res[,-1] <- sapply(C5.0Tree_res[,-1], function(x) as.numeric(as.character(x)))
C5.0Tree_res$AUC <- (C5.0Tree_res$sens + C5.0Tree_res$spec)/2
C5.0Tree_res$type <- fct_explicit_na(C5.0Tree_res$type)

summaryresults(C5.0Tree_res, "c50.csv")

C50.plot <- gather(C5.0Tree_res[,-c(2,5:8)], "meas", "value", 2:4)

#PLOT
plot.data <- cbind(C50.plot, sparseLDA.plot[,3], rf.plot[,3], glmnet.plot[,3])
colnames(plot.data) <- c("type", "meas", "c50", "sLDA", "RF", "GLMNET")
plot.data <- gather(plot.data, "model", "value", 3:6)

bwplot(model ~ value | meas , data = plot.data[which(plot.data$type == "normal"),], xlab = "")

## EVALUATE THE STABILITY MEASURES ####
stab <- c(getStability(glmnet_nogue_norm)$stability, getStability(rf_nogue_norm)$stability,
  getStability(C5.0Tree_nogue_norm)$stability, getStability(sparseLDA_nogue_norm)$stability,
  getStability(glmnet_nogue_rose)$stability, getStability(rf_nogue_rose)$stability,
  getStability(C5.0Tree_nogue_rose)$stability, getStability(sparseLDA_nogue_rose)$stability,
  getStability(glmnet_nogue_smote)$stability, getStability(rf_nogue_smote)$stability,
  getStability(C5.0Tree_nogue_smote)$stability, getStability(sparseLDA_nogue_smote)$stability)

stab <- matrix(stab, ncol = 4, byrow = TRUE)
colnames(stab) <- c("glmnet", "rf", "c50", "sLDA")
row.names(stab) <- c("None", "ROSE", "SMOTE")

stab_j <- c(stabilityJaccard(listfeats(glmnet_features$norm)), stabilityJaccard(listfeats(rf_features$norm)),
  stabilityJaccard(listfeats(C5.0Tree_features$norm)), stabilityJaccard(listfeats(sparseLDA_features$norm)),
  stabilityJaccard(listfeats(glmnet_features$rose)), stabilityJaccard(listfeats(rf_features$rose)),
  stabilityJaccard(listfeats(C5.0Tree_features$rose)), stabilityJaccard(listfeats(sparseLDA_features$rose)),
  stabilityJaccard(listfeats(glmnet_features$smote)), stabilityJaccard(listfeats(rf_features$smote)),
  stabilityJaccard(listfeats(C5.0Tree_features$smote)), stabilityJaccard(listfeats(sparseLDA_features$smote)))

stab_j <- matrix(stab_j, ncol = 4, byrow = TRUE)
colnames(stab_j) <- c("glmnet", "rf", "c50", "sLDA")
row.names(stab_j) <- c("None", "ROSE", "SMOTE")

stab_o <- c(stabilityOchiai(listfeats(glmnet_features$norm)), stabilityOchiai(listfeats(rf_features$norm)),
            stabilityOchiai(listfeats(C5.0Tree_features$norm)), stabilityOchiai(listfeats(sparseLDA_features$norm)),
            stabilityOchiai(listfeats(glmnet_features$rose)), stabilityOchiai(listfeats(rf_features$rose)),
            stabilityOchiai(listfeats(C5.0Tree_features$rose)), stabilityOchiai(listfeats(sparseLDA_features$rose)),
            stabilityOchiai(listfeats(glmnet_features$smote)), stabilityOchiai(listfeats(rf_features$smote)),
            stabilityOchiai(listfeats(C5.0Tree_features$smote)), stabilityOchiai(listfeats(sparseLDA_features$smote)))

stab_o <- matrix(stab_o, ncol = 4, byrow = TRUE)
colnames(stab_o) <- c("glmnet", "rf", "c50", "sLDA")
row.names(stab_o) <- c("None", "ROSE", "SMOTE")

##FORMAL TEST of stability
hypothesisTestCompareStabilities(C5.0Tree_nogue_norm, rf_nogue_norm)
hypothesisTestCompareStabilities(C5.0Tree_nogue_rose, C5.0Tree_nogue_smote)

hypothesisTestCompareStabilities(glmnet_nogue_norm, glmnet_nogue_rose)
hypothesisTestCompareStabilities(glmnet_nogue_norm, glmnet_nogue_smote)
hypothesisTestCompareStabilities(glmnet_nogue_smote, glmnet_nogue_rose)

hypothesisTestCompareStabilities(C5.0Tree_nogue_norm, C5.0Tree_nogue_smote)
hypothesisTestCompareStabilities(C5.0Tree_nogue_norm, C5.0Tree_nogue_rose)
hypothesisTestCompareStabilities(C5.0Tree_nogue_smote, C5.0Tree_nogue_rose)

## plot
corrplot(stab, method="shade", is.corr=FALSE, 
         cl.lim = c(0, 1), tl.col = "black",
         addCoef.col = "black")

corrplot(stab_j, method="shade", is.corr=FALSE, 
         cl.lim = c(0, 1), tl.col = "black",
         addCoef.col = "black")

corrplot(stab_o, method="shade", is.corr=FALSE, 
         cl.lim = c(0, 1), tl.col = "black",
         addCoef.col = "black")

# Boxplot

comp_stabmeas <- as.data.frame(t(cbind(stab,stab_j,stab_o)))
comp_stabmeas <- cbind(comp_stabmeas, c(rep("Noguiera", 4), rep("Jaccard", 4), rep("Ochiai", 4)))
comp_stabmeas <- gather(data = comp_stabmeas, "rebal", "value", 1:3)
colnames(comp_stabmeas)[1] <- "stab"

bp <- ggplot(comp_stabmeas, aes(x=stab, y=value, fill=rebal)) + 
  geom_boxplot()+
  labs(x="Stability indices", y = "Stability", fill = "Rebalancing")

bp + scale_fill_brewer(palette="Blues") + theme_classic()

# Similar boxplot for accuracy

bp <- ggplot(plot.data, aes(x=meas, y=value, fill=type)) + 
  geom_boxplot()+
  labs(x="Accuracy measure", y = "", fill = "Rebalancing")

bp + scale_fill_brewer(palette="Blues") + theme_classic()


## CALCULATE THE NUMBER OF SELECTED FEATURES ####
n_features <- data.frame(type = glmnet_nfeatures$type, 
                         glmnet = glmnet_nfeatures$n,
                         RF = rf_nfeatures$n,
                         sLDA = sparseLDA_nfeatures$n,
                         C50 = C5.0Tree_nfeatures$n)


n_features <- n_features[-1,]
str(n_features)
n_features[,2:5] <- sapply(n_features[,2:5], function(x) as.numeric(as.character(x)))


tab <- n_features %>%
  group_by(type) %>%
  summarise_if(is.numeric, list(~mean(., na.rm = TRUE), ~sd(., na.rm = TRUE), ~median(., na.rm = TRUE), ~min(., na.rm = TRUE), ~max(., na.rm = TRUE)), na.rm = TRUE)
write.csv2(tab, "nfeat.csv")

# Similar boxplot for nfeat

n_features_long <- gather(n_features, "model", "value", 2:5)

bp <- ggplot(n_features_long, aes(x=model, y=value, fill=type)) + 
  geom_boxplot()+
  labs(x="Model", y = "Number of features", fill = "Rebalancing")

bp + scale_fill_brewer(palette="Blues") + 
  theme_classic() #+
  #facet_zoom(ylim = c(60, 70))

## CLINICAL INTERPRETATION

## GLMNET
output <- matrix(unlist(glmnet_features$norm), nrow = 10, byrow = FALSE)
sel_feat <- matrix(NA, nrow = dim(testX)[2], ncol = 10)
row.names(sel_feat) <- colnames(testX)

for (i in 1:10) {
  tmp <- table(output[i,])
  id <- match(names(tmp),row.names(sel_feat))
  
  sel_feat[id,i] <- tmp
  
}

sel_feat[which(is.na(sel_feat))] <- 0

row.names(sel_feat) <- 1:nrow(sel_feat)
corrplot(sel_feat, method="shade", is.corr=FALSE, tl.col = "black",
         tl.cex = 0.6, cl.pos = "r", cl.align.text = "l", cl.cex = 0.6,
         cl.lim = c(0,100))

colnames(testX)[c(68, 48, 69, 34,47)]

## Most often in top 10
tab <- rowSums(sel_feat)
barplot(tab[order(tab, decreasing = T)], las = 2, horiz=FALSE, cex.names=0.6)

## RF

output <- matrix(unlist(rf_features$norm), nrow = 10, byrow = FALSE)
sel_feat <- matrix(NA, nrow = dim(testX)[2], ncol = 10)
row.names(sel_feat) <- colnames(testX)

for (i in 1:10) {
  tmp <- table(output[i,])
  id <- match(names(tmp),row.names(sel_feat))
  
  sel_feat[id,i] <- tmp
  
}

sel_feat[which(is.na(sel_feat))] <- 0

row.names(sel_feat) <- 1:nrow(sel_feat)
corrplot(sel_feat, method="shade", is.corr=FALSE, tl.col = "black",
         tl.cex = 0.6, cl.pos = "r", cl.align.text = "l", cl.cex = 0.6,
         cl.lim = c(0,100))

colnames(testX)[c(68,69,48)]

## C50

output <- matrix(unlist(C5.0Tree_features$norm), nrow = 10, byrow = FALSE)
sel_feat <- matrix(NA, nrow = dim(testX)[2], ncol = 10)
row.names(sel_feat) <- colnames(testX)

for (i in 1:10) {
  tmp <- table(output[i,])
  id <- match(names(tmp),row.names(sel_feat))
  
  sel_feat[id,i] <- tmp
  
}

sel_feat[which(is.na(sel_feat))] <- 0

row.names(sel_feat) <- 1:nrow(sel_feat)
corrplot(sel_feat, method="shade", is.corr=FALSE, tl.col = "black",
         tl.cex = 0.6, cl.pos = "r", cl.align.text = "l", cl.cex = 0.6,
         cl.lim = c(0,100))

colnames(testX)[c(68,69,48)]

## Most often in top 10: GLMNET: ROSE vs SMOTE vs nothing

## None
output <- matrix(unlist(glmnet_features$norm), nrow = 10, byrow = FALSE)
sel_feat <- matrix(NA, nrow = dim(testX)[2], ncol = 10)
row.names(sel_feat) <- colnames(testX)

for (i in 1:10) {
  tmp <- table(output[i,])
  id <- match(names(tmp),row.names(sel_feat))
  
  sel_feat[id,i] <- tmp
  
}

sel_feat[which(is.na(sel_feat))] <- 0

row.names(sel_feat) <- 1:nrow(sel_feat)
corrplot(sel_feat, method="shade", is.corr=FALSE, tl.col = "black",
         tl.cex = 0.6, cl.pos = "r", cl.align.text = "l", cl.cex = 0.6,
         cl.lim = c(0,50))

## ROSE
output <- matrix(unlist(glmnet_features$rose), nrow = 10, byrow = FALSE)
sel_feat <- matrix(NA, nrow = dim(testX)[2], ncol = 10)
row.names(sel_feat) <- colnames(testX)

for (i in 1:10) {
  tmp <- table(output[i,])
  id <- match(names(tmp),row.names(sel_feat))
  
  sel_feat[id,i] <- tmp
  
}

sel_feat[which(is.na(sel_feat))] <- 0

row.names(sel_feat) <- 1:nrow(sel_feat)
corrplot(sel_feat, method="shade", is.corr=FALSE, tl.col = "black",
         tl.cex = 0.6, cl.pos = "r", cl.align.text = "l", cl.cex = 0.6,
         cl.lim = c(0,50))

## SMOTE
output <- matrix(unlist(glmnet_features$smote), nrow = 10, byrow = FALSE)
sel_feat <- matrix(NA, nrow = dim(testX)[2], ncol = 10)
row.names(sel_feat) <- colnames(testX)

for (i in 1:10) {
  tmp <- table(output[i,])
  id <- match(names(tmp),row.names(sel_feat))
  
  sel_feat[id,i] <- tmp
  
}

sel_feat[which(is.na(sel_feat))] <- 0

row.names(sel_feat) <- 1:nrow(sel_feat)
corrplot(sel_feat, method="shade", is.corr=FALSE, tl.col = "black",
         tl.cex = 0.6, cl.pos = "r", cl.align.text = "l", cl.cex = 0.6,
         cl.lim = c(0,50))
