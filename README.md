# coursera-practical-machine-learning

---
title: "Practical machine learning"
author: "Menno Bremer"
date: "9-12-2018"
output:
  html_document: https://github.com/mennobremer93/coursera-practical-machine-learning.git
  pdf_document: default
---

# Practical Machine Learning project

### Menno Bremer
### December 9, 2018

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

## Load data

```{r setup, include=FALSE}
setwd("~/Documents/Coursera")
install.packages("caret")
install.packages("ggplot2")
install.packages("randomForest")
```

```{r}
library("caret")
library("ggplot2")
library("randomForest")

```

```{r}
train <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
test <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))

summary(train)
summary(train$classe)
```

## Split training and test data

Set aside a subset of the training data for cross validation (70%)

```{r}
Training <- createDataPartition(y=train$classe, p=0.7, list=FALSE)
inTrain <- train[Training, ]
inTest <- train[-Training, ]
dim(inTrain)
```

```{r}
dim(inTest)
```

## Feature Selection

Second, transform the data, include only the variables we want to use to built the prediction model. 
Remove variables with near zero variance, missing data, and variables which are useless as predictors.

```{r}
intrainselection <- inTrain
for (i in 1:length(inTrain)) {
  if (sum(is.na(inTrain[ , i])) / nrow(inTrain) >= .7) {
    for (j in 1:length(intrainselection)) {
      if (length(grep(names(inTrain[i]), names(intrainselection)[j]))==1) {
        intrainselection <- intrainselection[ , -j]
      }
    }
  }
}

dim(intrainselection)
```

```{r}
#remove columns that are obviously not predictors
intrains <- intrainselection[,8:length(intrainselection)]

#remove variables with near zero variance
NZV <- nearZeroVar(intrains, saveMetrics = TRUE)
NZV 
```

```{r}
keep <- names(intrains)

```

## Random Forest Model

Random forest model is used to build the Machine Learning algorithm as it should be more accurate than most other models based on information from the lectures.

First, fit the model on the training data.

```{r}
set.seed(151)

fitmod <- randomForest(classe~., data = intrains)
print(fitmod)
```

### Out of sample error
Second, use the model to predict the variable classe on the subset of testing data (cross validation).

```{r}
install.packages("e1071")
```

```{r}
library("e1071")
```

```{r}
predict1 <- predict(fitmod, inTest, type = "class")
confusionMatrix(inTest$classe, predict1)
```

### in sample error

```{r}
predicttrain <- predict(fitmod, inTrain, type = "class")
confusionMatrix(inTrain$classe, predicttrain)
```

Accuracy of in sample is lower than out of sample (99.63% vs 100%). 

## Model (test set)

```{r}
predict3 <- predict(fitmod, test, type = "class")
print(predict3)
```

### Reference  
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative 
Activity Recognition of Weight Lifting Exercises. Proceedings of 4th 
International Conference in Cooperation with SIGCHI (Augmented Human '13).
Stuttgart, Germany: ACM SIGCHI, 2013.

