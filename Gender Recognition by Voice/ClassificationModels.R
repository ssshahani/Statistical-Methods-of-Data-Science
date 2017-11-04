# Gender Recognition By Voice
# Authors: Fangzheng Sun, Huimin Ren,Sahil Shahani,Shanhao Li,Yun Yue

library(pls)
library(glmnet)
library(class)
library(MASS)
library(tree)
library(randomForest)
library(gbm)
library(e1071)
library(caret)

# Split data to 75% training and 25% test dataset. 
# In case of data snooping, we trained and tested our models with 75% training dataset
# and used 25% test dataset in the final step to test our best method.
#setwd("F:/MA543/FinalProjectCode/train&test.R")
source("train&test.R")

voice.train = train_set()
voice.test = test_set() 
pca.d = 1 # difference with best PCA components.
cv.i = 5 # how many times for cross Validation to split data

# second dividing (validation set and hyper-parameter set)
# data split
DataSplit = function(train.data){
  train_whole_index = 1:dim(train.data)[1]
  validation_index = sample(dim(train.data)[1], dim(train.data)[1]*0.75)
  voice.validation = train.data[validation_index, ]
  voice.HP = train.data[-validation_index, ]
  return(list("voice.validation" = voice.validation,"voice.HP" = voice.HP))
}  

# bootstrap
bootstrap = function(n,train.data) {
  voice.train.index = rownames(train.data)
  voice.train.bs.index = sample(voice.train.index,replace = TRUE)
  voice.train.bs = train.data[voice.train.bs.index,]
  return(voice.train.bs)
}

# pca
PCA = function(n,train.data){
  voice.train.bs <- bootstrap(n,train.data)
  predictors = voice.train.bs[,-length(train.data)] 
  pca <- prcomp(predictors, center = TRUE, scale. = TRUE) 
  pca.summary <- summary(pca)
  
  # find the first number of components whose cumulative contribution rate is more than 95%  
  cumu.prop = rep(0,length(predictors))
  cumu.prop[pca.summary$importance[3,]>=0.95]=1
  t <- match(1,cumu.prop) #the first PCA whose number of components is < 0.95
  return(list("pca" = pca,"t" = t,"train.data" = voice.train.bs))
}

##############################################################################
# Logistic Regression (with PCA)
LG.PCA = function(){
  pca = PCA(1,voice.train)
  lg.pca <- pca$pca
  #t <- pca$t
  voice.train.bs.lg <- pca$train.data
  # 
  error.avg.pca = c()
  #table.info = c() # this stores the predicted class and truee class for all iterations
  #for (j in (t-pca.d):(t+pca.d )){
  #  pca.x = lg.pca$x[,1:j]
  #  voice.train.bs.lg.pca = cbind(data.frame(pca.x),voice.train.bs.lg$label)
  #  colnames(voice.train.bs.lg.pca)[j+1] <- "label"
  j = 0  
    error = c()
    gender.HP.sum = c()
    lg.pred.sum = c()
    for(i in 1:cv.i){
      data = DataSplit(voice.train.bs.lg)
      voice.validation <- data$voice.validation
      voice.HP <- data$voice.HP
      
      sub.voice.fit.lg <- glm(label~., family = binomial, data=voice.validation)
      sub.voice.pre.lg <- predict(sub.voice.fit.lg, voice.HP, type = "response")
      
      sub.voice.pre.lg.result <- rep("male", length(sub.voice.pre.lg))
      sub.voice.pre.lg.result[sub.voice.pre.lg < 0.5] <- "female"
      
      error[i] <- mean(sub.voice.pre.lg.result != voice.HP$label)
      gender.HP.sum = c(gender.HP.sum, voice.HP[, ncol(voice.HP)])
      lg.pred.sum = c(lg.pred.sum, sub.voice.pre.lg.result)
    }
    
    error.avg.pca[j] <- mean(error)
    j = j+1
    gender.HP.sum = ifelse(gender.HP.sum == 2, "male", "female")
    table.info = cbind(table.info, gender.HP.sum, lg.pred.sum)
 # }
  min.error.pca.lg <- min(na.omit(error.avg.pca))
  index = which.min(na.omit(error.avg.pca)) 
  confusion_matrix = table(table.info[, index*2 - 1], table.info[, index*2])
  best.pca <- match(min.error.pca.lg,error.avg.pca)
  #cat(sprintf("Logistic Regression: \nbest principle components: %s, 
  #            minimum error: %.4f\n"
  #            ,best.pca,min.error.pca.lg))
  return(list("best.pca" = best.pca, "error" = min.error.pca.lg,
              "confusion.matrix" = confusion_matrix))
}
LG.PCA()
##############################################################################
# KNN (with PCA)
KNN.PCA = function(){
  pca = PCA(1111,voice.train)
  knn.pca <- pca$pca
  t <- pca$t
  voice.train.bs.knn <- pca$train.data
  
  avg.error.pca = c()
  best.knn = c()
  table.info = c()
  for (j in (t-pca.d):(t+pca.d)){
    knn.x = knn.pca$x[,1:j]
    voice.train.bs.knn.pca = cbind(data.frame(knn.x),voice.train.bs.knn$label)
    colnames(voice.train.bs.knn.pca)[j+1] <- "label"
    
    # find the best k
    knn.model <- train(label~., data=voice.train.bs.knn.pca, method='knn',
                       tuneGrid=expand.grid(.k=1:25),metric='Accuracy',
                       trControl=trainControl(method='repeatedcv', number=10, repeats=10))
    best.knn[j] <- match(max(knn.model$result$Accuracy),knn.model$result$Accuracy)
    
    newlabel <- ifelse(voice.train.bs.knn$label == "male", 2,1)
    voice.train.bs.knn.pca = cbind(data.frame(knn.x),newlabel)
    colnames(voice.train.bs.knn.pca)[j+1] <- "newlabel"
    
    error = c()
    gender.HP.sum = c()
    knn.pred.sum = c()
    for (i in 1:cv.i){
      data = DataSplit(voice.train.bs.knn.pca)
      voice.validation <- data$voice.validation
      voice.HP <- data$voice.HP
      
      sub.voice.pre.knn <- knn(voice.validation, voice.HP, 
                                voice.validation$newlabel, k=best.knn[j])
      error[i] <- mean(sub.voice.pre.knn != voice.HP$newlabel)
      gender.HP.sum = c(gender.HP.sum, voice.HP[, ncol(voice.HP)])
      knn.pred.sum = c(knn.pred.sum, sub.voice.pre.knn)
      
    }
    avg.error.pca[j] <- mean(error)
    table.info = cbind(table.info, gender.HP.sum, knn.pred.sum)
  }
  
  min.error.pca.knn <- min(na.omit(avg.error.pca)) # omit nan value
  # since the list combination outputs "2" as "male" and "1" as "female", we have to 
  # convert them back to the original classes
  table.info = ifelse(table.info == 2, "male", "female")
  index = which.min(na.omit(avg.error.pca)) 
  
  confusion_matrix = table(table.info[, index*2 - 1], table.info[, index*2])
  
  best.pca <- match(min.error.pca.knn,avg.error.pca)
  cat(sprintf("knn best k: %s, best principle components: %s, 
              minimum error: %.4f\n",
              best.knn[best.pca],best.pca, min.error.pca.knn))
  return(list("best.pca" = best.pca,"k" = best.knn[best.pca],
              "error" = min.error.pca.knn,
              "confusion.matrix" = confusion_matrix))
}

##############################################################################
# Decision Tree (with PCA)
Tree.PCA = function(){
  pca = PCA(22,voice.train)
  tree.pca <- pca$pca
  t <- pca$t
  voice.train.bs.tree <- pca$train.data
  
  table.info = c() # this stores the predicted class and truee class for all iterations
  avg.error.pca = c()
  for (j in (t-pca.d):(t+pca.d)){
    dt.x = tree.pca$x[,1:j]
    voice.train.bs.tree.pca = cbind(data.frame(dt.x),voice.train.bs.tree$label)
    colnames(voice.train.bs.tree.pca)[j+1] <- "label"
    
    error = c()
    gender.HP.sum = c()
    tree.pred.sum = c()
    for(i in 1:cv.i) {
      data = DataSplit(voice.train.bs.tree.pca)
      voice.validation = data$voice.validation
      voice.HP = data$voice.HP
      
      tree.voice.PCA = tree(label~., voice.validation)
      tree.pred.PCA = predict(tree.voice.PCA,voice.HP,type = "class")
      
      error[i] = mean(tree.pred.PCA != voice.HP$label)
      gender.HP.sum = c(gender.HP.sum, voice.HP[, ncol(voice.HP)])
      tree.pred.sum = c(tree.pred.sum, tree.pred.PCA)
    }
    avg.error.pca[j] <- mean(error)
    table.info = cbind(table.info, gender.HP.sum, tree.pred.sum)
  }
  
  min.error.pca.tree <- min(na.omit(avg.error.pca)) # omit nan value
  # since the list combination outputs "2" as "male" and "1" as "female", we have to 
  # convert them back to the original classes
  table.info = ifelse(table.info == 2, "male", "female")
  index = which.min(na.omit(avg.error.pca)) 
  confusion_matrix = table(table.info[, index*2 - 1], table.info[, index*2])
  best.pca <- match(min.error.pca.tree,avg.error.pca)
  cat(sprintf("Decision Tree: \nbest principle components: %s, 
              minimum error: %.4f\n"
              ,best.pca,min.error.pca.tree))
  return(list("best.pca" = best.pca,"error" = min.error.pca.tree, 
              "confusion.matrix" = confusion_matrix))
}

###################################################################################
# Random Forest (with PCA)
RF.PCA = function(){
  pca = PCA(204,voice.train)
  bg.pca <- pca$pca
  t <- pca$t
  voice.train.bs.bg <- pca$train.data
  
  error.avg.pca = c()
  table.info = c() # this stores the predicted class and truee class for all iterations
  for (j in (t-pca.d):(t+pca.d)){
    bg.x = bg.pca$x[,1:j]
    voice.train.bs.bg.pca = cbind(data.frame(bg.x),voice.train.bs.bg$label)
    colnames(voice.train.bs.bg.pca)[j+1] <- "label"
    
    error = c()
    gender.HP.sum = c()
    bg.pred.sum = c()
    for(i in 1:cv.i) {
      data = DataSplit(voice.train.bs.bg.pca)
      voice.validation <- data$voice.validation
      voice.HP <- data$voice.HP
      
      sub.voice.fit.bg <- randomForest(label~.,data = voice.validation,
                                       mtryin = 4,importance = TRUE)
      
      sub.voice.pre.bg <- predict(sub.voice.fit.bg, voice.HP)
      error[i] <- mean(sub.voice.pre.bg != voice.HP$label)
      gender.HP.sum = c(gender.HP.sum, voice.HP[, ncol(voice.HP)])
      bg.pred.sum = c(bg.pred.sum, sub.voice.pre.bg)
    }
    error.avg.pca[j] <- mean(error)
    table.info = cbind(table.info, gender.HP.sum, bg.pred.sum)
  }
  min.error.pca.bg <- min(na.omit(error.avg.pca))
  # since the list combination outputs "2" as "male" and "1" as "female", we have to 
  # convert them back to the original classes
  table.info = ifelse(table.info == 2, "male", "female")
  index = which.min(na.omit(error.avg.pca)) 
  confusion_matrix = table(table.info[, index*2 - 1], table.info[, index*2])
  best.pca <- match(min.error.pca.bg,error.avg.pca)
  
  cat(sprintf("Random Forest:\nbest principle components: %s, 
              minimum error: %.4f\n",
              best.pca, min.error.pca.bg))#
  return(list("best.pca" = best.pca, "error" = min.error.pca.bg, 
              "confusion.matrix" = confusion_matrix))
}

###################################################################################
# boosting (with PCA) (done)
Boost.PCA = function(){
  pca = PCA(502,voice.train)
  boost.pca <- pca$pca
  t <- pca$t
  voice.train.bs.boost <- pca$train.data
  
  fitControl <- trainControl(method = "repeatedcv",classProbs = T,
                             number = 5,summaryFunction = twoClassSummary,
                             repeats = 5)
  gbmGrid <-  expand.grid(interaction.depth = c(1,2,3,4),
                          n.trees = (1:10)*50,shrinkage = 0.001,
                          n.minobsinnode=10)
  
  error.avg.pca = c()
  best.depth = c()
  best.ntrees = c()
  table.info = c() # this stores the predicted class and truee class for all iterations
  for (j in (t-2):(t+2)){
    j = 10
    pca.x = boost.pca$x[,1:j]
    voice.train.bs.boost.pca = cbind(data.frame(pca.x),voice.train.bs.boost$label)
    colnames(voice.train.bs.boost.pca)[j+1] <- "label"
    
    gbmFit <- train(label ~ ., data = voice.train.bs.boost.pca
                    ,method = "gbm",trControl = fitControl,verbose = FALSE,
                    tuneGrid = gbmGrid,metric = "ROC")
    roc = gbmFit$results$ROC
    best.depth[j] <- gbmFit$results[match(max(roc),roc),]$interaction.depth
    best.ntrees[j] <- gbmFit$results[match(max(roc),roc),]$n.trees
    
    newlabel <- ifelse(voice.train.bs.boost$label == "male", 1,0)
    voice.train.bs.boost.pca = cbind(data.frame(pca.x),newlabel)
        
    error = c()
    gender.HP.sum = c()
    boost.pred.sum = c()
    for(i in 1:5) {
      i = 1
      data = DataSplit(voice.train.bs.boost.pca)
      voice.validation <- data$voice.validation
      voice.HP <- data$voice.HP
      
      sub.voice.fit.boost <- gbm(formula = newlabel ~ ., data = voice.validation, 
                                 shrinkage=0.001, distribution="bernoulli", 
                                 interaction.depth = best.depth[j], n.trees=best.ntrees[j])
      sub.voice.pre.boost <- predict(sub.voice.fit.boost, voice.HP, n.trees=best.ntrees[j])
      sub.voice.pre.boost.result <- rep(1, length(sub.voice.pre.boost))
      sub.voice.pre.boost.result[sub.voice.pre.boost <= 0] <- 0
      
      error[i] = mean(sub.voice.pre.boost.result != voice.HP$newlabel)
      gender.HP.sum = c(gender.HP.sum, voice.HP[, ncol(voice.HP)])
      boost.pred.sum = c(boost.pred.sum, sub.voice.pre.boost.result)
    }
    error.avg.pca[j] <- mean(error)
    table.info = cbind(table.info, gender.HP.sum, boost.pred.sum)
  }
  min.error.pca.boost <- min(na.omit(error.avg.pca))
  # since the list combination outputs "2" as "male" and "1" as "female", we have to 
  # convert them back to the original classes
  table.info = ifelse(table.info == 1, "male", "female")
  index = which.min(na.omit(error.avg.pca)) 
  confusion_matrix = table(table.info[, index*2 - 1], table.info[, index*2])
  best.pca <- match(min.error.pca.boost,error.avg.pca)
  cat(sprintf("Boost: best depth %s,best tree numbers %s\nbest principle components: %s, 
              minimum error: %.4f\n",best.depth[best.pca],best.ntrees[best.pca],best.pca,
              min.error.pca.boost))
  return(list("best.pca" = best.pca, "error" = min.error.pca.boost,
              "best.depth"=best.depth[best.pca],"best.n.trees"=best.ntrees[best.pca],
              "confusion.matrix" = confusion_matrix))
}

###################################################################################
# SVM (with PCA)
SVM.PCA = function(){
  pca = PCA(402,voice.train)
  svm.pca <- pca$pca
  t <- pca$t
  voice.train.bs.svm <- pca$train.data
  
  avg.error.pca1 = c()
  avg.error.pca2 = c()
  avg.error.pca3 = c()
  avg.error.pca4 = c()
  table.info = c()
  for (j in (t-pca.d):(t+pca.d)){
    svm.x = svm.pca$x[,1:j]
    voice.train.bs.svm.pca = cbind(data.frame(svm.x),voice.train.bs.svm$label)
    colnames(voice.train.bs.svm.pca)[j+1] <- "label"
    
    error.kernal1 = c()
    error.kernal2 = c()
    error.kernal3 = c()
    error.kernal4 = c()
    gender.HP.sum = c()
    svm1.pred.sum = c()
    svm2.pred.sum = c()
    svm3.pred.sum = c()
    svm4.pred.sum = c()
    for (i in 1:cv.i){
      data = DataSplit(voice.train.bs.svm.pca)
      voice.validation <- data$voice.validation
      voice.HP <- data$voice.HP
      gender.HP.sum = c(gender.HP.sum, voice.HP[, ncol(voice.HP)])
      
      sub.voice.fit.svm1 = tune(svm,label~., data=voice.validation,
                                kernel="linear")
      sub.voice.pre.svm1 = predict(sub.voice.fit.svm1$best.model, voice.HP)
      error.kernal1[i] = mean(sub.voice.pre.svm1 != voice.HP$label)
      svm1.pred.sum = c(svm1.pred.sum, sub.voice.pre.svm1)
      
      sub.voice.fit.svm2 = tune(svm,label~., data=voice.validation,
                                kernel="polynomial")
      sub.voice.pre.svm2 = predict(sub.voice.fit.svm2$best.model, voice.HP)
      error.kernal2[i] = mean(sub.voice.pre.svm2 != voice.HP$label)
      svm2.pred.sum = c(svm2.pred.sum, sub.voice.pre.svm2)
      
      sub.voice.fit.svm3 = tune(svm,label~., data=voice.validation,
                                kernel="radial",ranges=list(cost=10^(-2:3)), 
                                gamma=c(.5,1,2,4))
      sub.voice.pre.svm3 = predict(sub.voice.fit.svm3$best.model, voice.HP)
      error.kernal3[i] = mean(sub.voice.pre.svm3 != voice.HP$label)
      svm3.pred.sum = c(svm3.pred.sum, sub.voice.pre.svm3)
      
      sub.voice.fit.svm4 = tune(svm,label~., data=voice.validation,
                                kernel="sigmoid",ranges=list(cost=10^(-2:3)), 
                                gamma=c(.5,1,2,4))
      sub.voice.pre.svm4 = predict(sub.voice.fit.svm4$best.model, voice.HP)
      error.kernal4[i] = mean(sub.voice.pre.svm4 != voice.HP$label)
      svm4.pred.sum = c(svm4.pred.sum, sub.voice.pre.svm4)
      
    }
    avg.error.pca1[j] <- mean(error.kernal1)
    avg.error.pca2[j] <- mean(error.kernal2)
    avg.error.pca3[j] <- mean(error.kernal3)
    avg.error.pca4[j] <- mean(error.kernal4)
    table.info = cbind(table.info, gender.HP.sum, svm1.pred.sum, 
                       svm2.pred.sum, svm3.pred.sum, svm4.pred.sum)
  }
  min.error.pca.svm1 <- min(na.omit(avg.error.pca1)) # omit nan value
  min.error.pca.svm2 <- min(na.omit(avg.error.pca2))
  min.error.pca.svm3 <- min(na.omit(avg.error.pca3))
  min.error.pca.svm4 <- min(na.omit(avg.error.pca4))
  indexes = c(which.min(na.omit(avg.error.pca1)), which.min(na.omit(avg.error.pca1)),
              which.min(na.omit(avg.error.pca1)), which.min(na.omit(avg.error.pca1)))
  
  min.error.pca.svm.all <- c(min.error.pca.svm1,min.error.pca.svm2,
                             min.error.pca.svm3,min.error.pca.svm4)
  avg.error.pca.all<- list(avg.error.pca1, avg.error.pca2, avg.error.pca3, avg.error.pca4)
  
  min.error.pca.svm <- min(min.error.pca.svm.all)
  index_kernel = which.min(min.error.pca.svm.all)
  index_PC = indexes[index_kernel]
  # since the list combination outputs "2" as "male" and "1" as "female", we have to 
  # convert them back to the original classes
  table.info = ifelse(table.info == 2, "male", "female")
  confusion_matrix = table(table.info[, (5*(index_PC-1)+1)], table.info[, (5*(index_PC-1)+1+index_kernel)])
  best.svm.id <- match(min.error.pca.svm, min.error.pca.svm.all) # min error
  
  best.pca <- match(min.error.pca.svm.all[best.svm.id],avg.error.pca.all[[best.svm.id]]) # number of components
  
  svm.kernel = c("linear","polynomial","radial","sigmoid")
  
  cat(sprintf("svm best kernal: %s, best principle components: %s, 
              minimum error: %.4f\n",
              svm.kernel[best.svm.id],best.pca, 
              min.error.pca.svm.all[best.svm.id]))
  return(list("best.pca" = best.pca,"kernal" = svm.kernel[best.svm.id],
              "error" = min.error.pca.svm.all[best.svm.id], 
              "confusion.matrix" = confusion_matrix))
}

########################################################################################
########################################################################################
# find the best model with smallest error
lg = LG.PCA()
knn = KNN.PCA()
tree = Tree.PCA()
rf = RF.PCA()
boost = Boost.PCA()
svm1 = SVM.PCA()

lg.error <- lg$error
knn.error <- knn$error
tree.error <- tree$error
rf.error <- rf$error
boost.error <- boost$error
svm1.error <- svm1$error

lg.cm <- lg$confusion.matrix
lg.cm
knn.cm <- knn$confusion.matrix
knn.cm
tree.cm <- tree$confusion.matrix
tree.cm
rf.cm <- rf$confusion.matrix
rf.cm
boost.cm <- boost$confusion.matrix
boost.cm
svm1.cm <- svm1$confusion.matrix
svm1.cm

model <- c("LG","KNN","Tree","Random Forest","Boost","SVM")

error.all <- c(lg.error,knn.error,tree.error,
               rf.error,boost.error,svm1.error)
# compare the avg_error for each method and find the best model
error.all
# Considering that KNN and Random Forest had similar error, 
# we tested these two methods with voice.test
########################################################################################
########################################################################################
# test voice.test
# PCA for train and test data set
#source("C:/Users/rhm22/OneDrive/WPI/2017Spring/DS502/Project/code/train&test.R")
voice.train = train_set()
voice.test = test_set() 

predictors.voice = voice_data[,-length(voice_data)]
pca.voice <- prcomp(predictors.voice, center = TRUE, scale. = TRUE) 

#############################################################################
knn.train.x = pca.voice$x[train_index, 1:knn$best.pca]
newlabel <- ifelse(voice.train$label == "male", 2,1)
train.knn = cbind(data.frame(knn.train.x),newlabel)

knn.test.x = pca.voice$x[test_index, 1:knn$best.pca]
newlabel2 <- ifelse(voice.test$label == "male",2,1)
test.knn = cbind(data.frame(knn.test.x),newlabel2)

voice.pre.knn <- knn(train.knn, test.knn, train.knn$newlabel, k=knn$k) 
error.knn <- mean(voice.pre.knn != test.knn$newlabel)
error.knn
# Confusion Matrix
voice.pre.knn = ifelse(voice.pre.knn == 2, "male", "female")
test.knn$newlabel = ifelse(test.knn$newlabel == 2, "male", "female")
confusion_matrix.knn <- table(voice.pre.knn,test.knn$newlabel)
confusion_matrix.knn

###################################################################
# # RandomForest
# #train dataset
rf.train.x = pca.voice$x[train_index, 1:rf$best.pca]
train.rf = cbind(data.frame(rf.train.x),voice.train$label)
colnames(train.rf)[rf$best.pca+1] <- "label"

# test dataset
rf.test.x = pca.voice$x[test_index, 1:rf$best.pca]
test.rf = cbind(data.frame(rf.test.x),voice.test$label)
colnames(test.rf)[rf$best.pca+1] <- "label"

# model
fit.rf <- randomForest(label~.,data = train.rf,
                       mtryin = 4,importance = TRUE)
pre.rf <- predict(fit.rf, test.rf)
error.rf <- mean(pre.rf != test.rf$label)
error.rf

confusion_matrix.rf <- table(pre.rf,test.rf$label)
confusion_matrix.rf