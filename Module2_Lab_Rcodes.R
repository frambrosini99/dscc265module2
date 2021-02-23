### Lab Session on Module 2 - Linear Regression 2
### Subset selection, Ridge and Lasso Regression
### Cross-Validation Approach

## install packs first
install.packages("ggplot2") #high quality graphs
install.packages("lattice") #
install.packages("car") #
install.packages("glmnet") #fit a GLM with lasso, ridge or elasticnet regularization
install.packages("leaps") #subset selection

######
# Module 2 (ISLR's Chapter 6 Lab 1 and Lab 2 with modification)
## Subset Selection Methods

# Best Subset Selection

## import data set

library(ISLR)
fix(Hitters)
names(Hitters)
dim(Hitters) #322  20

# numerical-categorical variables
str(Hitters)
summary(Hitters)

# omit the na's (though this is not recommended in real situations!)
sum(is.na(Hitters$Salary)) #59 na's rows: sorry to loose these :(
Hitters=na.omit(Hitters)
dim(Hitters) #263  20

sum(is.na(Hitters)) #now, no na's in the data: be happy Joe!

# Salary is the response variable

## Then do descriptive plots, summary, corr matrix, pairs, EDA etc
# your portion

## leaps pack to use subset selection
# performs an exhaustive search for the best subsets of the variables in x for predicting y in linear regression, using an efficient branch-and-bound algorithm. It is a compatibility wrapper for regsubsets does the same thing better.
library(leaps)

## Full model: lm(y~.) and regsubsets(y~.) for best subset selection
?regsubsets #Model selection by exhaustive search, forward or backward stepwise, or sequential replacement

#nbest: number of subsets of each size to record
#nvmax: maximum size of subsets to examine
#method=c("exhaustive","backward", "forward", "seqrep")
# i want 5 best predictors in the model using forward selection method
# you can use different model selection criteria such as AIC, BIC...
# rss (SSE):	Residual sum of squares for each model
# Coefficients and the variance-covariance matrix for one or model models can be obtained with the coef and vcov methods.
# ISLR uses exhaustive (including all possibilities), here we use forward and call it regfit.m1 with 5 best
regfit.m1=regsubsets(Salary~., data=Hitters, nbest=1, 
                       nvmax=5, method="forward")
reg.summary=summary(regfit.m1)

# observe what star (*) means in forward method
reg.summary

names(regfit.m1) #call what you want
regfit.m1$method #method

names(reg.summary) #call what you want
reg.summary$adjr2 
reg.summary$rss #SSE for each model

coef(regfit.m1, 1:5) # use just 5
vcov(regfit.m1, 5)

###
##nvmax=19 - all predictors in the data set
regfit.m2=regsubsets(Salary~.,data=Hitters, nbest=1, 
                       nvmax=19, method="forward")

reg.summary=summary(regfit.m2)

# observe what star (*) means in forward method
reg.summary

names(reg.summary)
reg.summary$rsq

coef(regfit.m2, 1:19) #coefficients: use 19, not 1:19, see
vcov(regfit.m2, 19) #diagonals are var(estimate) so get se's here


## Visualizing the outputs in the model selection method
par(mfrow=c(2,2))
plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l")
plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points(11,reg.summary$adjr2[11], col="red",cex=2,pch=20)
plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp",type='l')
which.min(reg.summary$cp)
points(10,reg.summary$cp[10],col="red",cex=2,pch=20)
which.min(reg.summary$bic)
plot(reg.summary$bic,xlab="Number of Variables",ylab="BIC",type='l')
points(6,reg.summary$bic[6],col="red",cex=2,pch=20)

dev.off()

# ranked predictors according to the metrics below
?plot.regsubsets
plot(regfit.m2,scale="r2")
plot(regfit.m2,scale="adjr2")
plot(regfit.m2,scale="Cp")
plot(regfit.m2,scale="bic")

coef(regfit.m2,6)

### backward method: using all predictors
# interpret the result
regfit.bwd=regsubsets(Salary~.,data=Hitters,
                      nvmax=19,method="backward")

summary(regfit.bwd)


###
# Forward and Backward Stepwise Selection: we will use these
# Exhaustive is by default: it searches best subset
regfit.full=regsubsets(Salary~.,data=Hitters, nbest=1, 
                       nvmax=19, method="exhaustive")
summary(regfit.full)

regfit.fwd=regsubsets(Salary~.,data=Hitters,
                      nvmax=19,method="forward")
summary(regfit.fwd)

regfit.bwd=regsubsets(Salary~.,data=Hitters,
                      nvmax=19,method="backward")
summary(regfit.bwd)

# coeffs: first 7
coef(regfit.full,7)
coef(regfit.fwd,7)
coef(regfit.bwd,7)


###### Cross-Validation
##
## Choosing best Among Models using CV

# split the data into train and test
set.seed(99)
n = nrow(Hitters)
n
n_train = ceiling(.80 * n)
n_train

# choose randomly 80% 
train=sample(c(TRUE,FALSE), size=n, 
             prob=c(.80, .20), rep=TRUE) #randomly select index whether train or not from row
train
test=(!train)
test

dim(Hitters)

# see train and test data ready!
dim(Hitters[train,])
dim(Hitters[test,])


# let's fit using train data set, exhaustive search
regfit.best=regsubsets(Salary~.,data=Hitters[train,],nvmax=19)
regbest.summary = summary(regfit.best)
regbest.summary

regbest.summary$adjr2 #Adj-Rsq for each model. See 19 models.
regbest.summary$rss #SSE for each model. See 19 models.
regbest.summary$rss
  
coef(regfit.best, 19) #see coeff names and coeff estimates for 19th exhaused model

# this gets train MSE
# you can store Adj-Rsq, SSE, Cp etc
# i use here MSE
train.mat=model.matrix(Salary~., data=Hitters[train,])
dim(train.mat) #209  20

train.errors=rep(NA,19)
for(i in 1:19){
  coefi=coef(regfit.best,id=i)
  yhat=train.mat[,names(coefi)]%*%coefi
  train.errors[i]=mean((Hitters$Salary[train]-yhat)^2) #this gets train MSE
}

train.errors # for 19 models fitted, observe MSE's

which.min(train.errors)
coef(regfit.best,19)


# now, get test errors for each model built
# first, make model matrix of test data
test.mat=model.matrix(Salary~., data=Hitters[test,])
dim(test.mat) #54  20

# get MSE fro each model from test data
val.errors=rep(NA,19)
for(i in 1:19){
  coefi=coef(regfit.best,id=i)
  pred=test.mat[,names(coefi)]%*%coefi
  val.errors[i]=mean((Hitters$Salary[test]-pred)^2) #let's get test MSE
}

# now compare
# side-by-side plot on train and test errors (MSE's)
# Your portion: make sure the graph is readable, with title, x-y label, legends etc
cbind(train.errors, val.errors)
plot(1:19, train.errors,  col='blue')
par(new=TRUE)
plot(1:19, val.errors, col='red', add=TRUE)

# let's find best test MSE
which.min(val.errors)
coef(regfit.best,10) # see, 10-predictor (10th model) gives best MSE!! We find it!

# so far, feel free to redesign the code, make easier. step() is an alternative to use


########
### let's write own function to predict the results
### We will need in k-Fold, for each left fold to test

predict.regsubsets=function(object, newdata, id, ...){
  form=as.formula(object$call[[2]])
  mat=model.matrix(form,newdata)
  coefi=coef(object,id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi #prediction or fitted results
}

# using all data, best 10-variable model (model with 10 predictors) using best subset method
regfit2.best=regsubsets(Salary~.,data=Hitters,nvmax=19)
summary(regfit2.best)
coef(regfit2.best,10)

coef(regfit.best,10) # see, 10-predictor (10th model) gives best MSE!! We find it!
#observe these are not same estimates! why?


######### 
## k-Fold CV
## You feel free to use your way for efficient k-fold CV
k=10 #you will need k=5 in the assignment
set.seed(99)
folds=sample(1:k,nrow(Hitters),replace=TRUE)
folds

cv.errors=matrix(NA,k,19, dimnames=list(NULL, paste(1:19)))
cv.errors

# run best.fit before the k-fold loop: the code gets only (k-1) portion of data except first hold
best.fit=regsubsets(Salary~.,data=Hitters[folds!=1,],
                    nvmax=19, method = "exhaustive")
summary(best.fit)

best.fit=regsubsets(Salary~.,data=Hitters[folds!=1,],
                    nvmax=19, method = "backward")
summary(best.fit)

best.fit=regsubsets(Salary~.,data=Hitters[folds!=1,],
                    nvmax=19, method = "forward")
summary(best.fit)


k = 10
  
# now, looping/k-fold procedure
for(j in 1:k){
  # using 19 models obtained from (k-1)-fold data
  best.fit=regsubsets(Salary~.,data=Hitters[folds!=j,],
                      nvmax=19, method = "exhaustive")
  for(i in 1:19){
    # predict the held data for test MSE
    pred=predict(best.fit,Hitters[folds==j,],id=i)
    cv.errors[j,i]=mean( (Hitters$Salary[folds==j]-pred)^2)
  }
}

# MSE values for each model, for each held portion in k-fold
# Which model to use to calculate the test MSE?
# MSE_{test}
cv.errors

# how to calculate the averaged MSE_{test} for each model?
# for each model, see test MSE from k-fold
mean.cv.errors=apply(cv.errors,2,mean)
mean.cv.errors

# now, decide which model is best (optimal, smallest k-fold test MSE)


# how to calculate the averaged MSE_{train} for each model? asked in he assignment 
# you need to modify the loop, collect the MSEs from (k-1)-folds


# plot of test MSE
par(mfrow=c(1,1))
plot(mean.cv.errors,type='b')

reg.best=regsubsets(Salary~.,data=Hitters, nvmax=19)
coef(reg.best,10) #or 11


###### Shrinkage Methods in Regression
## Ridge Regression and the Lasso (ISLR's Chapter 6, Lab 2)

# design matrix (exclude 1's vector)
x=model.matrix(Salary~.,Hitters)[,-1]
View(x)
dim(x) #263  19
y=Hitters$Salary
hist(y)

# do EDA

# Ridge Regression
library(glmnet)
?glmnet #fit a GLM with lasso or elasticnet regularization

# grid for lambda
grid=10^seq(10,-2,length=100)
grid
min(grid);max(grid)

# fit ridge regression: x is data or data matrix
ridge.mod=glmnet(x,y,alpha=0,lambda=grid) #alpha=1 is the lasso penalty, and alpha=0 the ridge penalty.

dim(coef(ridge.mod))

ridge.mod$lambda
hist(ridge.mod$lambda)
ridge.mod$lambda[50]

coef(ridge.mod)[,50]
sqrt(sum(coef(ridge.mod)[-1,50]^2))

ridge.mod$lambda[60]
coef(ridge.mod)[,60]

sqrt(sum(coef(ridge.mod)[-1,60]^2))

# predict using lambda=50
predict(ridge.mod, s=50, type="coefficients")[1:20,]

##
set.seed(99)
train=sample(1:nrow(x), nrow(x)/2)
test=(-train)
y.test=y[test]
ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=grid, thresh=1e-12)
ridge.pred=predict(ridge.mod,s=4,newx=x[test,])

mean((ridge.pred-y.test)^2)
mean((mean(y[train])-y.test)^2)

ridge.pred=predict(ridge.mod,s=1e10,newx=x[test,])
mean((ridge.pred-y.test)^2)

ridge.pred=predict(ridge.mod,s=0,newx=x[test,],exact=T,x=x[train,],y=y[train])
mean((ridge.pred-y.test)^2)

# fit mlr
lm(y~x, subset=train)

# get estimated/fitted values
predict(ridge.mod,s=0, exact=T,
        type="coefficients",x=x[train,],y=y[train])[1:20,]

# 
set.seed(99)
?cv.glmnet #Does k-fold cross-validation for glmnet, produces a plot, and returns a value for lambda
cv.out=cv.glmnet(x[train,],y[train],alpha=0)
plot(cv.out)

bestlam=cv.out$lambda.min
bestlam # cross-validated lambda that yields best error

ridge.pred=predict(ridge.mod,s=bestlam,newx=x[test,])
mean((ridge.pred-y.test)^2) #test MSE associated with this best lambda above

# refit the model with optimal lambda (best)
out=glmnet(x,y,alpha=0)
predict(out,type="coefficients",s=bestlam)[1:20,] #none of coeffs are zero. Not GOOD!!


### Now, let's do more aggressive approach with L1 in feature elimination
# The Lasso

lasso.mod=glmnet(x[train,],y[train],alpha=1,lambda=grid)
plot(lasso.mod)

# perform CV
set.seed(99)
cv.out=cv.glmnet(x[train,],y[train],alpha=1)
plot(cv.out)

bestlam=cv.out$lambda.min
bestlam

lasso.pred=predict(lasso.mod,s=bestlam,newx=x[test,])
mean((lasso.pred-y.test)^2)

out=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:20,]
lasso.coef  # see many coeffs are zero!! We did!

lasso.coef
lasso.coef[lasso.coef!=0] # see features eliminated! feel free to use this best set for 
