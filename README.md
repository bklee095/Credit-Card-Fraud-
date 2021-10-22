
# Credit card fraud detection model accuracy performance comparison

# Dataset

[Data source URL](https://drive.google.com/file/d/1CTAlmlREFRaEN3NoHHitewpqAtWS5cVQ/view)

The dataset has 284,807 data entries and 31 attributes. The predictor variables are composed of 28 separate variables characterizing the transaction and the amount of the transaction. The response variable is a binary categorical variable showing whether the record is a fraud or not.

# Exploring Data

```{r}
library(DataExplorer)

plot_bar(df)
```
![image](https://user-images.githubusercontent.com/74638365/138370439-b75ab0a3-3b2c-4fe1-8fa1-948cefc7a18a.png)
<br/>
_Binary response variable distribution_

<br/>

```{r}
table(df$Class)
prop.table(table(df$Class))
```

   .       | Negative (0) | Positive (1)
------------|----------|---------
Count       | 284,315  | 492
Probability | 0.99827  | 0.001727

The dataset's response variable is extremely unbalanced. This is a common case in real-world applications. There are foten more negative cases than the positive cases when it comes to detection of special cases in most subjects. That may be the reason why machine learning is deployed at the first place: in order to look through plethora of data and detecting special cases (medical condition detection, financial fraud case detection, etc.)

<br/>
<br/>
<br/>

```{r}
plot_intro(df)
```
![image](https://user-images.githubusercontent.com/74638365/138371062-78774e1d-626b-40c2-922d-620020cf2e74.png)
<br/>
_High level exploratory data analysis_

<br/>

No missing values found in the dataset.

<br/>
<br/>
<br/>

```{r}
# Time is irrelevant to the data analysis
carddf = df[, -c(1)]
```


# Data Analysis

```{r}
# Setting a random seed for reproducibility
set.seed(100)

# 80-20 split for random (sample()) training and testing set
division = sort(sample(nrow(carddf), nrow(carddf) * 0.80))

training = carddf[division,]
testing = carddf[-division,]
```

```{r}
# Maintaining the binary response variable ratio in training and testing sets

table(training$Class)
table(testing$Class)
```

. | Negative (0) | Positive (1)
--|----------|---------
Training | 227,447 | 398
Testing | 56,868 | 94


<br/>
<br/>
<br/>

### Logistic Regression

```{r}
# Training a logistic regression model to the training set
lr = glm(Class~.,
         training,
         family = binomial)

# Prediction against the testing set based on the new model
model.probability = predict(lr, 
                            newdata = testing, 
                            type = "response")
```

```{r}
prediction = rep("0", 56962)
prediction[model.probability > 0.5] = "1"

table(prediction, testing$Class)
```

Prediction | 0 | 1
-------|-------|------
0 | 56,857 | 34
1 | 11 | 60

Overall accuracy = (TP+TN)/(TP+FP+TN+FN) = 0.9992

<br/>
<br/>
<br/>

However, general accuracy isn't the best performance evaluation metrics in this situation. Over 99.8% of the case are negative. Hypothetically speaking, if the model were to have concluded negative for all of the cases, it would have still gotten over 0.998 overall accuracy.


```{r}
library(pROC)
prediction = as.numeric(prediction)
auc(testing$Class, prediction)
```
Area under the curve (AUC) score = 0.8191, showing stark contrast to the overall accuracy value.


<br/><br/><br/>

```{r}
library(ROCR)
ROCRpred = prediction(predictTrain, training$Class)
```

```{r}
ROCRperf = performance(ROCRpred, "tpr", "fpr")
```

```{r}
# Plot ROC curve
plot(ROCRperf, colorize = TRUE, 
     main = "LogReg ROC Curve",
     print.cutoffs.at = seq(0, 1, by = 0.1))
abline(coef = c(0,1))
```
![image](https://user-images.githubusercontent.com/74638365/138372326-4e08d81b-5bb3-4a52-a178-e0e4cda6e61f.png)
<br/>
_Logistic Regression model ROC curve with thresholds_

<br/>

Changing the threshold of the logistic regression model would be ineffective in this case.



<br/><br/><br/>

### Principal Component Analysis (PCA)

With too many attributes and data entries, PCA can be performed to see if dimensionality reduction is possible.

```{r}
PCA = prcomp(carddf[, -ncol(carddf)], scale = TRUE)
summary(PCA)
```
![PCA table](https://user-images.githubusercontent.com/74638365/138372594-6872f238-206f-491e-afc4-99b8ffad3e9c.PNG)
_Principal Component Analysis - Importance of components table_

PCA is ineffective for this dataset as none of the principal components show dominant variance explanation.

<br/><br/><br/>

### Random Forest Classifier

```{r}
library(randomForest)

RFmodel = randomForest(Class~., 
                       data = training,
                       ntree = 100,
                       mtry = 4)
```

Random forest classifier trained with the training set.

100 trees and subset of 4 random variables per tree.

<br/>

```{r}
varImpPlot(RFmodel)
```
![image](https://user-images.githubusercontent.com/74638365/138372672-fca781c7-a23a-4d0a-93da-a26480fd411f.png)

Variable importance plot shows which variables are the most important in prediction processes. Effective for variable selection, but irrelevant for now.

```{r}
p = predict(RFmodel, testing)
mean(p == testing$Class)
```
Overall accuracy = 0.9995

Random forest yields comparable general detection accuracy performance with the logistic regression. 

```{r}
pred_prob = predict(RFmodel, testing, type = "prob")

auc = auc(testing$Class, pred_prob[ ,2])
auc
plot(roc(testing$Class, pred_prob[ ,2]), main = "RF ROC Curve")
```

Area under the curve (AUC) score = 0.9344

![image](https://user-images.githubusercontent.com/74638365/138372929-5042e762-e7a8-4c2d-a806-99cd9b736606.png)
<br/>
_Random Forest model ROC curve_

But a significantly higher AUC score.


<br/>
<br/>
<br/>
<br/>

### Conclusion

ROC AUC score refers to the area under the ROC curve. It shows the probability that a randomly chosen positive data entry will have a higher rank than a randomly chosen negative data entry for the given dataset. For this reason, ROC AUC score is often preferred over general accuracy metric for binary classification settings. 

Random forest classifier yielded a higher detection accuracy, proving its worth and showing its robustness as a emsemble method.
