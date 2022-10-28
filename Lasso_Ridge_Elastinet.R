#Lasso or Ridge

library(glmnet)

x=model.matrix(campaign_response~.-1,data=train_df) 
y=df_train$campaign_response
cv.lasso=cv.glmnet(x,y,family="binomial",type.measure = "deviance",alpha=1) 
cv.ridge=cv.glmnet(x,y,family="binomial",type.measure = "deviance",alpha=0) 
cv.elastinet=cv.glmnet(x,y,family="binomial",type.measure = "deviance",alpha=0.5) 

coef.cv.glmnet(cv.lasso,s="lambda.1se")

plot(cv.lasso, xvar="lambda")
