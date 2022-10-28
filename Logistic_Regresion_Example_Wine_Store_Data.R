# Classification with Logistic Regression
# Code Snippets for Propensity Model
# rajesh.parvathin@tigeranalytics.com

# Clearing Environment
rm(list=ls())

# Setting working directory
# setwd(choose.dir(default = "", caption = "Select folder"))
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Output goes to file below as well as console
sink("Logistic_Regresion_Example_Wine_Store_Data.txt", split=TRUE)

# Setting Environment
options(stringsAsFactors = TRUE)

# Libraries
library(dplyr) 
library(data.table)
library(tidyverse)
library(ggplot2)
library(glmnet) # Generalized Linear models
# library(rpart) # package for CART
# library(randomForest)
library(mltools) # Used for one-hot encoding (Dummy variable creation)
library(pROC) # For Plotting ROC Curves
# library(plotROC)
# library(car) # Companion to Applied Regression


## Scoping Questions for any project 
### Problem Statement - Interpretable model (Logistic/CART) vs. Blackbox model (Random Forest / Boosting)
### How the model will be used - Will be used decide performance measure and business metric 
### Assumptions if any

## Modeling Steps

### Data Exploration and Data Cleaning
### Check Assumptions
### Create Training, Test and holdout/vault samples

#PDE functsions
#Bivariate Analysis functions
source('./PDE_RLibrary/TigerPDE.R')

# Classification Functions
# Will add these as functions in later version
# Performance Measures - 
# Misclassification Matrix, Precision/Hit Rate, Recall/Capture Rate 
# ROC Curve and AUC 
# Gains Chart
# source('ClassificationFunctions.R')


#---- Read Data -----------
# df <- read.csv(file.choose(), sep = ",", header = T, quote = "")
df <- read.csv('./../Data/wine_store_model_data.csv')
df_backup<-df

# df<-df_backup

# Part 1: EDA of variables 

### Data Exploration and Data Cleaning
# Data is already clean so no cleaning

# Let's Understand the data
# Univariate Analysis 

View(PDE.variable_summary(df))
View(PDE.numeric_summary(df))

# Density plots
setwd("./PDE_RLibrary")

PDE.density_plots(df)

# Checking for Outliers 
# View(sapply(df[,sapply(df, is.numeric)], function(x) quantile(x, c(.01,.05,.25,.5,.75,.90,.95, .99, 1),na.rm=TRUE) ))

# num_cols = colnames(df)[sapply(df, is.numeric)]
# num_cols<-num_cols[-1]
# for(v in num_cols) {
#   hist(df[,v])
# }

# Bivariate Analysis 

df$campaign_response_f<-as.factor(df$campaign_response)
num_variables <- names(df)[sapply(df,is.numeric)]
View(num_variables)
num_variables<-num_variables[-1] #Removing xid

PDE.bivariate_plots(df,num_variables,c("campaign_response_f"))
df$campaign_response_f<-NULL

PDE.percentile_plots(df)

# for (i in num_variables) {
# eda.binary_y_cont_x_bins(df,i,"campaign_response")
# }

# Divide dataset into train, test and vault 60, 20 and 20 split
set.seed(2019)
split_names<-c("train", "test", "vault")
split_probs<-c(0.6, 0.2, 0.2)
split_flag<-sample(split_names,nrow(df),replace=T,prob=split_probs)
table(split_flag)


# Creating dummy variables for categorical variables
df$occupation<-as.factor(df$occupation)
df$gender<-as.factor(df$gender)
df$region<-as.factor(df$region)

df <- one_hot(as.data.table(df))


# Removing one of the levels in categorical from dummies
View(PDE.variable_summary(df))

df$gender_M<-NULL
df$region_Urban<-NULL
df$occupation_Student<-NULL

View(PDE.variable_summary(df))

df$xid<-NULL

model_run <- function(){
  train_df<-df[split_flag=="train",]
  test_df<-df[split_flag=="test",]
  vault_df<-df[split_flag=="vault",]
  
  # Plain Vanilla Logistic Model
  # Include all variables
  
  model1 <- glm(campaign_response~.,data = train_df, family = binomial)
  
  # Interpret the results of the logistic regression
  print(summary(model1))
  
  # Accuracy Measures
  train_prob <- predict(model1,train_df,type="response")
  summary(train_prob)
  pred_train<-(train_prob>0.5)*1
  print(table(train_df$campaign_response,pred_train))
  
  message("Accuracy")
  print(mean(train_df$campaign_response == pred_train)) # Accuracy 
  message("Precision / Hit Rate")
  print(sum(train_df$campaign_response * pred_train) / sum(pred_train)) # Precision / Hit Rate
  message("Recall / Capture Rate")
  print(sum(train_df$campaign_response * pred_train) / sum(train_df$campaign_response)) # Recall / Capture Rate
  
  # 2 Class Imbalance
  # Change the probability Threshold and show how the accuracy measures change
  #What if I changethe probability threshold 
  pred_train<-(train_prob>0.6)*1
  table(train_df$campaign_response,pred_train)
  
  mean(train_df$campaign_response == pred_train) # Accuracy 
  sum(train_df$campaign_response * pred_train) / sum(pred_train) # Precision / Hit Rate
  sum(train_df$campaign_response * pred_train) / sum(train_df$campaign_response) # Recall / Capture Rate
  
  
  #What if I changethe probability threshold 
  pred_train<-(train_prob>0.4)*1
  table(train_df$campaign_response,pred_train)
  
  mean(train_df$campaign_response == pred_train) # Accuracy 
  sum(train_df$campaign_response * pred_train) / sum(pred_train) # Precision / Hit Rate
  sum(train_df$campaign_response * pred_train) / sum(train_df$campaign_response) # Recall / Capture Rate
  
  
  # How is the accuracy in Test Sample
  test_prob <- predict(model1,test_df,type="response")
  summary(test_prob)
  pred_test<-(test_prob>0.5)*1
  message("Test Confusion Matrix")
  print(table(test_df$campaign_response,pred_test))
  mean(test_df$campaign_response == pred_test)
  
  test_accuracy<-mean(test_df$campaign_response == pred_test)
  test_precision<-sum(test_df$campaign_response * pred_test) / sum(pred_test)
  test_recall<- sum(test_df$campaign_response * pred_test) / sum(test_df$campaign_response)
  
  message("Accuracy in Test Sample")
  message("Accuracy")
  print(test_accuracy)
  message("Precision / Hit Rate")
  print(test_precision)
  message("Recall / Capture Rate")
  print(test_recall)
  
  # Is there a way to measure performance that doesn't depend on probability threshold 
  
  # ROC curve, AUC
  plot(roc(test_df$campaign_response, test_prob), print.thres = c(.1, .5), col = "red", print.auc = T)
  # What is ROC Curve
}

model_run()

# Creating the predicted test data frame
test_campaign_response<-test_df$campaign_response
test_pred <- data.frame(cbind(test_campaign_response, test_prob))

# Ordering the dataset in descending probability
test_pred <- test_pred[order(1-test_pred$test_prob),]

# Creating the cumulative density
test_pred$cumden <- cumsum(test_pred$test_campaign_response)/sum(test_pred$test_campaign_response)

# Creating the % of population
test_pred$perpop <- (seq(nrow(test_pred))/nrow(test_pred))*100

# Gains Chart
plot(test_pred$perpop,test_pred$cumden,type="l",xlab="% of Population",ylab="% of Responders")

# 3 Interactions
# Try all Second order interactions and see what sticks

model2_all_interactions <- glm(campaign_response~.^2,data = train_df, family = binomial)
summary(model2_all_interactions)

# Let's look at the performance

# Accuracy Measures
train_prob <- predict(model2_all_interactions,train_df,type="response")
summary(train_prob)
pred_train<-(train_prob>0.5)*1
table(train_df$campaign_response,pred_train)

message("Accuracy")
mean(train_df$campaign_response == pred_train) # Accuracy 
message("Precision / Hit Rate")
sum(train_df$campaign_response * pred_train) / sum(pred_train) # Precision / Hit Rate
message("Recall / Capture Rate")
sum(train_df$campaign_response * pred_train) / sum(train_df$campaign_response) # Recall / Capture Rate

# Looks like second model is doing a better job

# How about Accuracy in Test Sample

test_prob <- predict(model2_all_interactions,test_df,type="response")
summary(test_prob)
pred_test<-(test_prob>0.5)*1
table(test_df$campaign_response,pred_test)
mean(test_df$campaign_response == pred_test)


test_accuracy<-mean(test_df$campaign_response == pred_test)
test_precision<-sum(test_df$campaign_response * pred_test) / sum(pred_test)
test_recall<- sum(test_df$campaign_response * pred_test) / sum(test_df$campaign_response)

message("Accuracy in Test Sample")
message("Accuracy")
test_accuracy
message("Precision / Hit Rate")
test_precision
message("Recall / Capture Rate")
test_recall

plot(roc(test_df$campaign_response, test_prob), print.thres = c(.1, .5), col = "red", print.auc = T)

# View(PDE.numeric_summary((train_df)))
# write.csv(names(train_df),"num_vars.csv")

# 4 Mix of Variables - Categorical and Continuous IDVs
# Can I use state variable as coninuous variable?

train_df %>% 
  select(state, campaign_response) %>% 
  group_by(state) %>%
  summarise_all(mean) %>% 
  View()

df$state<-as.factor(df$state)

df <- one_hot(as.data.table(df))

View(PDE.variable_summary(df))

df$state_99<-NULL #  Base State for Dummies

# Let's re-run the plain vanilla model again
model_run()
# Go back to line 131 - run code from line 131 to line 191

# 1 Multicollinearity

num_variables <- names(df_backup)[sapply(df_backup,is.numeric)]
corrmat<-cor((df_backup[num_variables]))
write.csv(corrmat,"correlation_matrix.csv")
class(train_df)
# Feature Engineering
# Does it make sense that charges are Intuitive 

summary(model1)
cor(train_df$in_store_wine_eq_volume,train_df$in_store_wine_dollar_sales)

df = df %>% 
  mutate(
    in_store_wine_price = (in_store_wine_dollar_sales / (in_store_wine_eq_volume+1)),
  	in_store_beer_price = (in_store_beer_dollar_sales / (in_store_beer_eq_volume+1)),
	  in_store_spirits_price = (in_store_spirits_dollar_sales/(in_store_spirits_eq_volume+1)),
	  delivery_wine_price = (delivery_wine_dollar_sales / (delivery_wine_eq_volume+1)),
	  delivery_beer_price = (delivery_beer_dollar_sales / (delivery_beer_eq_volume+1)),
	  delivery_spirits_price = (delivery_spirits_dollar_sales/(delivery_spirits_eq_volume+1))
  )

# Let's re-run the model again
model_run()
# Go back to line 131 - run code from line 131 to line 191
summary(model1)

# 6 Non-linear / Specification Problems
# 7 Piecewise Linear

select_vars<-c("age",
               "direct_emails",
               "direct_mails",
               "campaign_response")

df<-as.data.frame(df)
new_df<-df[select_vars]

new_df$Age_Deciles <- ntile(new_df$age, 10)

# new_df %>% 
#   select(Age_Deciles, age) %>% 
#   group_by(Age_Deciles) %>%
#   summarise_all(min) %>% 
#   View()

new_df %>% 
  select(Age_Deciles, campaign_response) %>% 
  group_by(Age_Deciles) %>%
  summarise_all(mean) %>% 
  plot()

new_df %>% 
  select(Age_Deciles, campaign_response) %>% 
  group_by(Age_Deciles) %>%
  summarise_all(mean) %>% 
  View()

new_df$direct_emails_bin <- ntile(new_df$direct_emails, 10)

new_df %>% 
  select(direct_emails_bin, campaign_response) %>% 
  group_by(direct_emails_bin) %>%
  summarise_all(mean) %>% 
  plot()

new_df$direct_mails_bin <- ntile(new_df$direct_mails, 10)

new_df %>% 
  select(direct_mails_bin, campaign_response) %>% 
  group_by(direct_mails_bin) %>%
  summarise_all(mean) %>% 
  plot()

# 8 Feature Selection
stepwise_selection<-step(model1,direction = c("both"))
summary(stepwise_selection)
summary(logistic_model1)
betas1<-logistic_model1$coefficients
betas_step<-stepwise_selection$coefficients


num_sum<-PDE.numeric_summary(df)
write.csv(num_sum,"num_sum.csv")
