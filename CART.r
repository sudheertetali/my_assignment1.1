#CART

cart_model_fit <- rpart(campaign_response ~ num_vars,method="class", data=train_df)
printcp(cart_model_fit) # display the results 
plotcp(cart_model_fit) 
plot(cart_model_fit,uniform=TRUE)
text(cart_model_fit, use.n=TRUE, all=TRUE, cex=.8)

best_cv_cp=0.010236
best_fit<-prune(cart_model_fit,cp=best_cv_cp)


