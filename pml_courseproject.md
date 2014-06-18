====================================
## Introduction
The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. The following report describes how 
we built our model and how we used cross validation dataset. We will also use our prediction model to predict 20 different test cases at last.

## Data preparation
1.Load the datasets and libraries.

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
# Load datasets
trainRawData <- read.csv("pml-training.csv", header = TRUE, stringsAsFactors = FALSE)
testRawData <- read.csv("pml-testing.csv", header = TRUE, stringsAsFactors = FALSE)
```


2.Divide the data into two sets: training (70%) and cross validation (30%). The test set is given so there is no need to generate from the code.

```r
set.seed(1234)
trainIndex <- createDataPartition(trainRawData$classe, p = 0.7, list = FALSE)
# training dataset
training = trainRawData[trainIndex, ]
# cross validation dataset
testing = trainRawData[-trainIndex, ]
```


3.Filter the numeric variables and remove all the missing values in these variables

```r
valid_idx = which(lapply(training, class) %in% "numeric")
validTrain <- preProcess(training[, valid_idx], method = "knnImpute")
combTrain <- cbind(training$classe, predict(validTrain, training[, valid_idx]))
combTest <- cbind(testing$classe, predict(validTrain, testing[, valid_idx]))

# testing dataset
validSubmission <- predict(validTrain, testRawData[, valid_idx])
```


4.Rename first Label as classe

```r
names(combTrain)[1] <- "classe"
names(combTest)[1] <- "classe"
```


## Prediction with Random Forest Model
Apply a random forest model to the numerical variables. 

```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
modFit <- randomForest(classe ~ ., combTrain)
modFit
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = combTrain) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 9
## 
##         OOB estimate of  error rate: 1.65%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3885   11    3    3    4    0.005376
## B   37 2587   28    3    3    0.026712
## C    1   29 2334   28    4    0.025876
## D    5    2   37 2203    5    0.021758
## E    3    5   11    4 2502    0.009109
```


### In-sample accuracy
In-sample accuracy is 100%, which shows that the model does not suffer from bias. 

```r
predictTrain <- predict(modFit, combTrain)
print(confusionMatrix(predictTrain, combTrain$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```


### Out-of-sample accuracy
The out-of-sample accuracy is 98.6%, which is enough to predict the 20 test observations.

```r
predictTest <- predict(modFit, combTest)
print(confusionMatrix(predictTest, combTest$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1666   11    2    2    2
##          B    5 1116   13    2    2
##          C    1   10 1000    5    7
##          D    1    0    9  952    3
##          E    1    2    2    3 1068
## 
## Overall Statistics
##                                         
##                Accuracy : 0.986         
##                  95% CI : (0.983, 0.989)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.982         
##  Mcnemar's Test P-Value : 0.48          
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.995    0.980    0.975    0.988    0.987
## Specificity             0.996    0.995    0.995    0.997    0.998
## Pos Pred Value          0.990    0.981    0.978    0.987    0.993
## Neg Pred Value          0.998    0.995    0.995    0.998    0.997
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.283    0.190    0.170    0.162    0.181
## Detection Prevalence    0.286    0.193    0.174    0.164    0.183
## Balanced Accuracy       0.996    0.988    0.985    0.992    0.993
```

 
## Test Set Prediction Results
Apply the model to the twenty test observations.

```r
answers <- predict(modFit, validSubmission)
answers
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

