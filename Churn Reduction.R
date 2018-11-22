  #Remove The Variable & Values In Environment
  rm(list=ls())
  
  #Set The Directory
  setwd("F:/Eddwisor/Task Program/Projects")
  
  #Get The Directory
  getwd()
  
  #------Load the Train dataset-------#
  train=read.csv("Train_data.csv",encoding = 'ISO-8859-1')
  
  #-----Check Dimension-------#
  dim(train)              #rows=3333 and column=21
  
  #Get the Column Names
  names(train)
  
  #Get the structure of the dataset
  str(train)
  head(train,4)
  
  #----------------------------------------------------------------------
  
    #No need of account length for churn reduction
    #get the index of column account length and remove it from the dataset
  
    account_length_index = match("account.length",names(train))
    account_length_index
    train=train[-account_length_index]
  #----------------------------------------------------------------------
  
  #After Removing The columns Check The train Dataset
    str(train)
  #Original dataset have 8 num variables, 7 int and 5 factor
  #----------------------------------------------------------------------
  
  #Seperate The Numeric & categorical variables In Actual Train Dataset
    
  #Seperate Numeric Variables :
  numeric_index=sapply(train,is.numeric)       #It shows only numeric value.
  numeric_index
  numeric_data=train[,numeric_index]           #Display only numeric data.
  numeric_data
  cnames=colnames(numeric_data)                #Numeric column names are stored in variable cnames.
  cnames

  #Seperate Categorical Variables :
  cat_index=sapply(train,is.factor)            #It shows only categorical value.
  cat_index
  cat_data=train[,cat_index]                   #Display only categorical data.
  cat_data
  cat_col_names=colnames(cat_data)             #Catgorical column names are stored in variable cat_col_cnames.
  cat_col_names
  
  #Now we have two subset of train dataset
  # 1. numeric_data which contains only numerical variables data
  # 2. cat_data which contains only categorical variables data
  
  #Check Dimension For Actual Dataset
  dim(train)                                   #rows=3333  & col=20
  
  #Check Dimension For Numeric Data
  dim(numeric_data)                            #rows=3333  & col=15
  
  #Check Dimension For Categorical Data
  dim(cat_data)                                #rows=3333  & col=5
  
  #----------------------------------------------------------------------
  #Churn is our target variable which is categorical
  #Let's check out if there exits target imbalance class problem 
  
  imbal_class=table(train$Churn)
  imbal_class                                  #False : 2850 $ True : 483
  barplot(table(train$Churn))                 
  #It can be seen here that 2850 elements are False and 483 elements are True
  #Thus we can say that target imbalance problem exists here
  #----------------------------------------------------------------------
  
  #calculating event rate
  483/3333                                     #14.49 %
  
  #----------------------------------------------------------------------
  #To solve these imbalance problem we used here simple sampling method
  set.seed(1234)
  sample_set =train[sample(nrow(train),size=2333,replace = TRUE),]
  sample_set
  
  barplot(table(sample(nrow(train),size=2333,replace=TRUE)))
  #----------------------------------------------------------------------
  #calculate number of observations in the sample
  (3333*70)/100                               #2333
  
  #After Applying Sampling Method The Result Will Be Given:
  imbal_class=table(sample_set$Churn)
  imbal_class                                 #So,Sample set contain FALSE : 1998 & TRUE : 335
  barplot(table(sample_set$Churn))       
  #---------------------------------------------------------------------
  #Calculate Churn event rate
  335/2333                                    #14.35%
  
  #14.35 % rate of churning on a sample set of 2333 observations
  #Now from here onwards we are having sample_set as our dataset to apply further processes over it.
  #---------------------------------------------------------------------
  ######################################################################
  
  #Onwards We are Using sample_set:
  #check The sample_set column names
  names(sample_set)
  
  #Seperate The Numeric & categorical variables In sample_set Dataset
  
  #Seperate Numerical Variables:
  sample_numeric_index=sapply(sample_set,is.numeric)              #It shows only numeric value.
  sample_numeric_index
  sample_numeric_data=sample_set[,sample_numeric_index]           #Display only numeric data.
  sample_numeric_data
  cnames=colnames(sample_numeric_data)                            #Numeric column names are stored in variable cnames.
  cnames
  
  #Seperate Categorical Variables:
  sample_cat_index=sapply(sample_set,is.factor)                   #It shows only categorical value.
  sample_cat_index
  sample_cat_data=sample_set[,sample_cat_index]                   #Display only categorical data.
  sample_cat_data
  sample_cat_col_names=colnames(sample_cat_data)                  #Catgorical column names are stores in variable sample_cat_col_cnames.
  sample_cat_col_names
  
  #Check Dimension Of Numerical Variables
   dim(sample_numeric_data)                                       #rows : 2333 & col : 15
  
  #Check Dimension of Categorical Variables
   dim(sample_cat_data)                                           #rows : 2333 & col : 5
   
  #------------------------------------------------------------------------
  #########################################################################
  
  #Bulid the Other Dataset For Performing Operations
  set.seed(1234)
  new_set =sample_set[sample(nrow(sample_set),size=1000,replace = TRUE),]
  new_set
  
  #Dimension of New Dataset i.e new_set
  dim(new_set)                                                   #rows : 1000 & col : 20
  
  #Check The Column Name
  names(new_set)
  
  #For Outlier Analysis We don't need of categorical variables 
  #as well some numerical variable so we drop that variable.
  
  #First Drop state:
  state_index=match("state",names(new_set))
  state_index
  new_set=new_set[-state_index]
  
  #Second Drop area_code:
  area_code_index = match("area.code",names(new_set))
  area_code_index
  new_set=new_set[-area_code_index]
  
  #Third Drop phone_number:
  phone_number_index=match("phone.number",names(new_set))
  phone_number_index
  new_set=new_set[-phone_number_index]
  
  #Fourth Drop international_plan:
  international_plan_index=match("international.plan",names(new_set))
  international_plan_index
  new_set=new_set[-international_plan_index]
  
  #Fifth Drop voice_mail_plan:
  voice_mail_plan_index=match("voice.mail.plan",names(new_set))
  voice_mail_plan_index
  new_set=new_set[-voice_mail_plan_index]
  
  #Sixth Drop number.customer.service.calls:
  number_customer_service_calls_index= match("number.customer.service.calls",names(new_set))
  number_customer_service_calls_index
  new_set=new_set[-number_customer_service_calls_index]
  
  #After Drop The variables Check the columns in new_Set
  names(new_set)                                          
  
  #Then Seperate Numerical & categorical Variables:
  
  #Seperate Numerical Variables:
  new_numeric_index=sapply(new_set,is.numeric)              #It shows only numeric value.
  new_numeric_index
  new_numeric_data=new_set[,new_numeric_index]              #Display only numeric data.
  new_numeric_data
  new_cnames=colnames(new_numeric_data)                     #Numeric column names are stored in variable new_cnames.
  new_cnames
  
  #Seperate Categorical Variables:
  new_cat_index=sapply(new_set,is.factor)                   #It shows only categorical value.
  new_cat_index
  new_cat_data=new_set[,new_cat_index]                      #Display only categorical data.
  new_cat_data
  new_cat_col_names=colnames(new_cat_data)                  #Catgorical column names are stores in variable new_cat_col_cnames.
  new_cat_col_names
  
  #####################################################################
  #--------------------------------------------------------------------
  
  #################Check For Missing Values Column Wise################
  
  sum(is.na(new_set))        # no any missing value is available
  
  #Plotting the Boxplot
  install.packages("caret")                   
  library(caret)
  
  for(i in 1:length(new_cnames))
  {
    assign(paste0("bx",i),ggplot(aes_string(y=(new_cnames[i]),x="Churn"),data=subset(new_set))+
             stat_boxplot(geom="errorbar",width=0.5)+
             geom_boxplot(outlier.colour="red",fill="grey",outlier.shape=18,outlier.size=1,notch="FALSE")+
             theme(legend.position="bottom")+
             labs(y=new_cnames[i],x="Churn")+
             ggtitle(paste("Box plot of Churn for",new_cnames[i])))
    print(i)
  }
  
  #Plotting Plots Together
  gridExtra::grid.arrange(bx10,bx4,ncol=2)
  gridExtra::grid.arrange(bx1,bx2,ncol=2)
  gridExtra::grid.arrange(bx8,bx7,ncol=2)
  gridExtra::grid.arrange(bx5,bx3,bx11,ncol=3)
  gridExtra::grid.arrange(bx12,bx13,ncol=2)
  gridExtra::grid.arrange(bx9,bx6,ncol=2)
  
  #Make a backup dataframe
  df=new_set   
  dim(new_set)                                                 #rows : 1000 & col : 14
  dim(df)                                                      #rows : 1000 & col : 14

  #Detect total number of outliers present in all of the numerical variables
  
  for(i in new_cnames)
  {
    print(i)
    val=new_set[,i][new_set[,i] %in% boxplot.stats(new_set[,i])$out]  #0
    new_set=new_set[which(!new_set[,i]%in% val),]              #889
  }
  
  #Check missing values
  outliers= apply(new_set, 2, function(x) sum(is.na(x)))
  outliers
  sum(is.na(new_set))                                          # sum(is.na(new_set)) = 0
  
  #Replace all outliers with NA and then impute   
  
  for(i in new_cnames)
  {
    val=new_set[,i][new_set[,i] %in% boxplot.stats(new_set[,i])$out]
    new_set[,i][new_set[,i] %in% val]=NA
  }
  
  #After applying imputation check outliers
  #Check Outliers
  outliers= apply(new_set, 2, function(x) sum(is.na(x)))
  outliers
  
  sum(is.na(new_set))                                          # 218

#For Removing Outliers we used imputed methods:
  
# 1.KNN imputation
install.packages("DMwR")
library(DMwR)
install.packages("VIM")
library("VIM")
new_set=kNN(new_set, k=3)
sum(is.na(new_set))

# 2.Mean
new_set_mean=mean(new_cnames)
sum(is.na(new_set_mean))

# 3.Median
new_set_median=median(new_cnames)
sum(is.na(new_set_median))

# 4.Mode                                       #Mode is apply on categorical variable
new_set_mode=mode(new_cat_data)
sum(is.na(new_set_mode))

###########################################################################

###########################Feature Selection################################

#Feature Selection is apply on sample_set data
#Feature Selection is solve by correlation as well as Chi-square Test.Correlation is apply only on Numeric variables
# & Chi-square Test is apply on categorical variables.

#Load the Package:
install.packages("corrgram")
library(corrgram)

#Correlation Plot
corrgram(sample_set[,sample_numeric_index],order=F,
         upper.panel = panel.pie ,text.panel = panel.txt,method="color",main="Correlation Plot")


#Chi-Square Test For Categorical Variable
factor_index=sapply(sample_set, is.factor)              #It shows categorical variable.
factor_index

factor_data=sample_set[,factor_index]                   #It shows Categorical Data.
factor_data

factor_cnames=colnames(factor_data)                     #It shows Categorical Column Names.
factor_cnames

#Here 5 are the categorical variables.
#(independent variable:4 & dependent variable:1)
for(i in 1:4)  
{
  print(names(factor_data)[i])
  print(chisq.test(table(factor_data$Churn,factor_data[,i])))
}
#Dimension Detection
train_deleted=subset(sample_set,select=-c(total.day.minutes,total.eve.minutes))
train_deleted

#############################################################################

#############################Decision Tree###################################
#For Decision Tree we used sample_set dataset.

#For Decision Tree model we need to load C50 library
install.packages("C50")
library(C50)

#Develop C50 model on sample_set dataset
C50_model=C5.0(Churn ~number.vmail.messages + total.day.minutes + total.day.calls + 
                 total.day.charge + total.eve.minutes + total.eve.calls + total.eve.charge +
                 total.night.minutes+total.night.calls+total.night.charge + 
                 total.intl.minutes + total.intl.calls +total.intl.charge  , data = sample_set)
summary(C50_model)

#For prediction we need test dataset.
#Load the test dataset
test=read.csv("Test_data.csv",encoding = 'ISO-8859-1')

#Predict the data for test dataset
C50_predictors=predict(C50_model,newdata = test,type="class")
C50_predictors

library(caret)
#Calculate the Performance of Model using C50 model

confMatrix_C50=table(test$Churn,C50_predictors)    #Accuracy = 90.22%
confusionMatrix(confMatrix_C50)

#Calculate False Negative Rate
FNR=FN/(FN+TP)        #142/142+82 = 63.39%

#Plot the decision tree
plot(C50_model, main = "Classification Tree")

#For Binary Tree model we need to load rpart library
library(rpart)
install.packages("rpart.plot")
library(rpart.plot)

#Develop CART model on sample_set dataset
cart_model = rpart(Churn ~number.vmail.messages + total.day.minutes + total.day.calls + total.day.charge + total.eve.minutes + total.eve.calls + total.eve.charge +total.night.minutes+total.night.calls+total.night.charge + total.intl.minutes + total.intl.calls +total.intl.charge, data=sample_set)
summary(cart_model)

#For prediction we need test dataset.
#Load the test dataset
test=read.csv("Test_data.csv",encoding = 'ISO-8859-1')

#Predict the data for test dataset
cart_predictors=predict(cart_model,newdata = test,type="class")
cart_predictors

#Calculate the Performance of Model using CART model
confMatrix_Cart=table(test$Churn,cart_predictors)    #Accuracy = 90.04%
confusionMatrix(confMatrix_Cart)

#Calculate False Negative Rate
FNR=FN/(FN+TP)        #151/151+73 = 67.41%

#Plot the tree using prp command defined in rpart.plot package
prp(cart_model, main = "Binary Tree")

#Decision Tree Using C50 Model
 # Accuracy : 90.22%
 # FNR : 63.39%

#Decision Tree Using CART Model
 # Accuracy : 90.04%
 # FNR : 67.41%
############################Random Forest######################################
#For Random Forest we used sample_set Datasets.

#Load The Package
library(randomForest)

#Create the Forest
RF_model<-randomForest(Churn ~ number.vmail.messages + total.day.minutes + total.day.calls + total.day.charge + total.eve.minutes + total.eve.calls + total.eve.charge +total.night.minutes+total.night.calls+total.night.charge + total.intl.minutes + total.intl.calls +total.intl.charge, data = sample_set)

# View the forest results.
print(RF_model) 

# Importance of each predictor.
out.importance <- round(importance(RF_model), 2)
print(out.importance )

#Predict the data for test dataset
randomForest_predictors=predict(RF_model,newdata = test,type="class")
randomForest_predictors

#Calculate the Performance of Model

confMatrix_randomForest=table(test$Churn,randomForest_predictors)    #Accuracy = 90.7%
confusionMatrix(confMatrix_randomForest)

#Calculate False Negative Rate
FNR=FN/(FN+TP)        #150/150+74 = 66.96%

#Plot the Random Forest
plot(RF_model , main = "Random Forest")

#Using Random Forest:
# Accuracy : 90.7%
# FNR : 66.96%

################Logistic Regression#####################
#For Logistic Regression We used sample_set dataset.
#Logistic m odel is apply on categorical data.
require(ISLR)
#Create a Logistic Model
logit_model<-glm(Churn ~state + international.plan + voice.mail.plan, data=sample_set ,family = binomial)
summary(logit_model)

#Predict the dataset for test data
logit_predictions=predict(logit_model,newdata = test,type="response")

#Convert into probabilities
logit_predictions=ifelse(logit_predictions>0.5, 1,0)
logit_predictions

#Calculate the performance of the model
confMatrix_logit=table(test$Churn,logit_predictions)  #Accuracy = (1409 + 32)/1667 = 86.44%
confMatrix_logit                                     

#Calculate False Negative Rate
FNR=FN/(FN+TP)        #192/(192+32) = 85.71%

#Plot the Logistic Regression
plot(logit_model)


#Using Logistic  Regression:
# Accuracy : 86.44%
# FNR : 85.71%

#Comparing within three models like Decision Tree,Random Forest and Logistic Regression,
#Three out of two model give the average result.
#Therefore we can select either of two models without loss of information.



