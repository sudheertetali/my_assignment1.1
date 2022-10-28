# Cross Validation 
nfolds<-10
folds=sample(rep(1:nfolds,length=nrow(train_df)))
table(folds)

cv.pred=matrix(NA,nrow(train_df),2) # case-by-case predicted and actual
indx <- 0

for(k in 1:nfolds){
  model_cv <- glm(campaign_response ~ . , family="binomial", data = train_df[folds!=k,keep_vars])
  
  validnx=train_df[folds==k,keep_vars]
  validnx$campaign_response<-NULL
  DPred <- predict(model_cv, validnx, type="response")
  
  cv.pred[(indx+1):(indx+nrow(validnx)),1] <- DPred
  cv.pred[(indx+1):(indx+nrow(validnx)),2] <- y[folds==k]
  
  indx <- indx+nrow(validnx)
  
  #Grid = table(y[folds==k],DPred)
  cat("\nFold")
  print(k)
}

# Model Stats and Other Diagnostics
# Confusion Matrix Precision, Recall
actual<-cv.pred[,2]
predicted_prob<-cv.pred[,1]
summary(predicted_prob)
summary(actual)

prob_threshold<-0.5
predicted<-(predicted_prob>=prob_threshold)*1
confusion_matrix<-table(actual,predicted)
confusion_vector <- as.vector(t(confusion_matrix))
model_precision<-(confusion_vector[4])/sum(confusion_vector)
model_recall<-(confusion_vector[1])/sum(confusion_vector)
accuracy<-(confusion_vector[1]+confusion_vector[4])/sum(confusion_vector)
#confusion_vector
cat("\nCross validation Model Metrics\n")

cat("\nconfusion matrix\n")
print(confusion_matrix)
cat("\nmodel precision\n")
print(model_precision)  

cat("\nmodel recall\n")
print(model_recall)  

cat("\nmodel accuracy\n")
print(accuracy)  

# Specificity, Sensitivity and AUC - ROC
library(pROC)
roc_obj <- roc(actual, predicted_prob)

#roc_df <- data.frame(TPR=rev(roc_obj$sensitivities),FPR=rev(1 - roc_obj$specificities))

# plot(roc_df$FPR,roc_df$TPR)
ggroc(roc_obj, legacy.axes = TRUE)
auc(roc_obj)
