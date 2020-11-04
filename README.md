# Understanding the importance of advert placements

## Introduction

The data analysed contains the number of impressions (advert views) on each website from various users, as well as whether or not they clicked on an advert. Using machine learning, I have shown how impressions from each website affect an individuals likelihood of clicking on an advert, thus allowing one to be able to reallocate their advertising spending in order to increase the number of advert clicks.

## 1 - Data review and pre-processing

Prior to analysis, the data had several issues, hence requiring some pre-processing. These issues and their mitigations are briefly noted below:

* A large number of missing values for impressions on one website. These were imputed using k-Nearest Neighbours.
* One extreme outlier. This was imputed with 0 (no impression).
* Outliers across most websites, defined by a Z-score greater than 3. These were left in, as their removal saw degredation to the predictive algorithms.
* Imbalanced dataset - only ~12% of the datapoints corresponded to an advert click. To manage this the models were trained with a subset of the data, which undersampled the non-clicks data. When validating the models, the testing dataset was representative of the real (imbalanced) data.

## 2 - Model selection and refinement

In order to understad the importance of impressions on different websites, I created several models to predict a users likelihood of clicking an advert, based on the impressions they have seen. These models were:

* Logistic regression.
* Random forest.
* k-Nearest Neighbours.

All models were trained and tested in 10 independent trials, the logistic regression performed best with mean accuracy, precision and recall of 0.95, 0.98, 0.70, respectively. The ROC curve for these models is shown below.

![](https://github.com/joebarnes1996/advertising/blob/master/images/model_roc_comparison.png)

Due to the strong performance of logistic regression, and that it is easily interpretted, I chose to use it as my final model to understand the effect of impressions on different websites on a users likelihood to clicking an advert. To refine the logistic regression model, I used forward feature selection to select the optimal combination of variables to use in prediction.

## 3 - Results and interpretations

To interpret the results, I briefly recap on logistic regression: <img src="https://render.githubusercontent.com/render/math?math=ln(\frac{P}{1-P}) = \sum \theta_i x_i">, where <img src="https://render.githubusercontent.com/render/math?math=\x_i"> and <img src="https://render.githubusercontent.com/render/math?math=\theta_i"> are the i-th variable (impressions on each website) and its respective coefficient. Note that here <img src="https://render.githubusercontent.com/render/math?math=\x_0 = 1"> and hence <img src="https://render.githubusercontent.com/render/math?math=\theta_0 x_0 = \theta_0 = constant">.

As one can see from the above, the probability of a user clicking an advert depends on the sum of the number of clicks for each website multiplied by a coefficient, hence, the greater a coefficient for a website, the larger the effect it has on a user clicking an advert. I do point out however, that this relationship is non-linear and should not be interpretted as such. The valueo of the coefficient for each website is shown in the below image.

![](https://github.com/joebarnes1996/advertising/blob/master/images/Feature_coefficients.png)

The above image shows the websites which have little and large effects on users' likelihoods of clicking on adverts. This information is useful for marketing teams, as it allows reallocation of advert expenditure. For example, one could remove any adverts they have on low impact websites such as Buddymedia, Docs, Diigo or Facebook and reallocate them to high impact websites such as Thisnext, Kaboodle or Mybloglog. Due to the non-linearity of the relationship of advert views to probability of clicks, this reallocation is simple, though it demonstrates that one can increase advert views with no additional expenditure.  
