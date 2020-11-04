# Understanding the importance of advert placements

## Introduction

The data analysed contains the number of impressions (advert views) on each website from various users, as well as whether or not they clicked on an advert. Using machine learning, I have shown how impressions from each website affect an individuals likelihood of clicking on an advert, thus allowing one to be able to reallocate their advertising spending in order to increase the number of advert clicks.

## Data review and pre-processing

Prior to analysis, the data had several issues, hence requiring some pre-processing. These issues and their mitigations are briefly noted below:

* A large number of missing values for impressions on one website. These were imputed using k-Nearest Neighbours.
* One extreme outlier. This was imputed with 0 (no impression).
* Outliers across most websites, defined by a Z-score greater than 3. These were left in, as their removal saw degredation to the predictive algorithms.
* Imbalanced dataset - only ~12% of the datapoints corresponded to an advert click. To manage this the models were trained with a subset of the data, which undersampled the non-clicks data. When validating the models, the testing dataset was representative of the real (imbalanced) data.

## Model selection and refinement

In order to understad the importance of impressions on different websites, I created several models to predict a users likelihood of clicking an advert, based on the impressions they have seen. These models were:

* Logistic regression.
* Random forest.
* k-Nearest Neighbours.

All models were trained and tested in 10 independent trials, the logistic regression performed best with mean accuracy, precision and recall of 0.95, 0.98, 0.70, respectively. The ROC curve for these models is shown below.

![](https://github.com/joebarnes1996/advertising/blob/master/images/model_roc_comparison.png)

Due to the strong performance of logistic regression, and that it is easily interpretted, I chose to use it as my final model to understand the effect of impressions on different websites on a users likelihood to clicking an advert. To refine the logistic regression model, I used forward feature selection to select the optimal combination of variables to use in prediction.

# Results and interpretations

