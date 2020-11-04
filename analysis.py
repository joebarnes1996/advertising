#=============================================================================

#=============================================================================
"""
1. Load libraries
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats




#=============================================================================

#=============================================================================
"""
2. Read the data
"""
# set the directory
path = r'C:\Users\joeba\github_projects\Advertising'
os.chdir(path)
data = pd.read_csv('data/data.csv')




#=============================================================================

#=============================================================================
"""
3. Inspect the data
"""
# check out the format - all websites should have numeric integer values
print(data.head())

# check for missing / non-numeric values
pd.options.display.max_rows = 85
print(data.dtypes)

# Diigo column is object, hence contains non integer values
# all Diigo values (even numbers) are strings
print(data[pd.to_numeric(data.Diigo, errors='coerce').isnull()].Diigo)
print(data.Diigo.value_counts())


# check for outliers - for now, remove columns with strings
data_out = data.drop(columns=['Unnamed: 0', 'Diigo'])
data_z = stats.zscore(data_out)
print('Out of all {} non-zero values, {} are considered outliers'
      .format(data_out[data_out> 0].count().sum(), len(data_out[data_z > 3])))
print('However, 0 is the absence of an advert. Hence it should not be'\
      'considered when finding outliers.')

# find columns with outliers when 0 is ignored
for i in data_out.columns[1:-1]:
    
    # get non-zero values for each column
    non_zero_values = data_out[data_out[i] != 0][i]
    z_score = stats.zscore(non_zero_values)
    
    print('\n\n\n{} has {} outliers, with a maximum value of {}'.
          format(i, len(z_score[z_score > 3]), non_zero_values.max()))


# what percentage of viewers clicked a link
print('{:.2f}% of viewers clicked on a link'.
      format(100 * len(data[data.Click == 1]) / len(data)))


#=============================================================================

#=============================================================================
"""
4. Pre-process the data
"""
# get rid of first column as it is not necessary
data = data.drop(columns='Unnamed: 0')


#==============================
# replace all missing values with NaN for easy replacement
data = data.replace('Error: value not found', np.nan)



#==============================
# replace NaN values with K-NN
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score as acc

# split the data, such as to be data with no NaN values and data with NaN values
# missing values in y_pred will be replaced by comparing to values in y_train
data_train = data[data.notnull().all(axis=1)]
data_pred  = data[data.isnull().any(axis=1)]

# for each column, replace missing values in data_pred using K-NN 
# fit to data_train
for i in ['Diigo']:

    # fit the knn model to the training data
    knn = KNN(n_neighbors=5)
    knn.fit(data_train.drop(columns=i), data_train[i]) # poss change to categorical
    
    # use knn to replace the missing values - assume nan values as 0 
    # when doing K-NN
    data_pred[i] = knn.predict(data_pred.drop(columns=i).replace(np.nan, 0))
    print(i)


# join the data back together
data = pd.concat([data_train, data_pred]).reset_index(drop=True)
data.Diigo = pd.to_numeric(data.Diigo)


# replace extreme value in Newsvine
data.Newsvine[data.Newsvine == data.Newsvine.max()] = np.median(data.Newsvine)

"""
Note - better model performance was found when the outliers of other columns 
       were ignored.
"""




#=============================================================================

#=============================================================================
"""
5. Do some exploratory analysis
"""
# get the mean adverts for clicks and no clicks
mean_no_clicks = data[data.Click == 0].mean(axis=0)
mean_clicks = data[data.Click == 1].mean(axis=0)


"""
Note - can see that for each type, more adverts appeared where there were
       clicks.
"""
plt.figure(figsize=(12, 6))
plt.grid()
plt.scatter(mean_clicks.index, mean_clicks, marker='.', label='Click')
plt.scatter(mean_clicks.index, mean_no_clicks, marker='.', label='No click')
plt.xticks(rotation=90)
plt.legend()
plt.ylabel('Mean number of adverts')
plt.savefig('images/mean_adverts_clicks')
plt.show()





#=============================================================================

#=============================================================================
"""
6. Create the initial models

Initially, will test:
    
    - logistic regression
    - random forest classifier
    - k-nearest neighbours

"""
# import asssesment stuff
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score as ps
from sklearn.metrics import recall_score as rs
from sklearn.preprocessing import StandardScaler as ss


# import models
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RFC



#==============================
# evaluate the three models over 10 trials by using accuracy, precision 
# and recall
accs_rfc = []
accs_lr = []
accs_knn = []

ps_rfc = []
ps_lr  = []
ps_knn = []

rs_rfc = []
rs_lr  = []
rs_knn = []


for i in range(10):
    
    #==============================
    # keep some data aside for testing
    X_train, X_test, y_train, y_test = tts(data[data.columns[:-1]], 
                                           data[data.columns[-1]], 
                                           test_size=0.25, 
                                           random_state=i)
    
    # reform the training data
    data_train = X_train
    data_train['Click'] = y_train
    
    # reform the testing data
    data_test = X_test
    data_test['Click'] = y_test
    
    
    
    #==============================
    # balance the training data to have equal clicks and no clicks so that the
    # data is not biased
    print('There are approx {:.2f} times more non-clicks than clicks.'
          .format(len(data_train[data_train.Click == 0]) / 
                  len(data_train[data_train.Click == 1]))
          )
    
    # get the clicks data
    data_train_clicks = data_train[data_train.Click == 1]
    data_train_no_clicks = data_train[data_train.Click == 0]
    
    
    # under sample the no clicks data - can just select the first N amount
    # as the train test split randomises the order
    """
    Note - oversampling was also trialled, but led to worse model performance
    """
    
    data_train_no_clicks = data_train_no_clicks.iloc[:len(data_train_clicks)]
    
    data_train = pd.concat([data_train_clicks, data_train_no_clicks])
    
    # get the training and testing data split
    X_train = data_train[data_train.columns[:-1]]
    X_test = data_test[data_test.columns[:-1]]
    
    y_train = data_train[data_train.columns[-1]]
    y_test = data_test[data_test.columns[-1]]
    

    #==============================
    # fit the models
    rfc = RFC().fit(X_train, y_train)
    lr = LR().fit(X_train, y_train)
    knn = KNN().fit(X_train, y_train)
    
    #make predictions
    y_pred_rfc = rfc.predict(X_test)
    y_pred_lr  = lr.predict(X_test)
    y_pred_knn = knn.predict(X_test)
    
    # get the metrics
    accs_rfc.append(acc(y_pred_rfc, y_test))
    accs_lr.append(acc(y_pred_lr,  y_test))
    accs_knn.append(acc(y_pred_knn, y_test))
    
    ps_rfc.append(ps(y_pred_rfc, y_test))
    ps_lr.append(ps(y_pred_lr, y_test))
    ps_knn.append(ps(y_pred_knn, y_test))
    
    rs_rfc.append(rs(y_pred_rfc, y_test))
    rs_lr.append(rs(y_pred_lr, y_test))
    rs_knn.append(rs(y_pred_knn, y_test))
       
    
    print(i)




#==============================
# examine performances of all models

"""
Note - can see that across all metrics, logistic regression performs best
"""

# accuracy
plt.figure(figsize=(12, 6))
plt.grid()
sns.distplot(accs_rfc, hist=False, kde_kws={"shade": True}, label='RFC')
sns.distplot(accs_lr,  hist=False, kde_kws={"shade": True}, label='LR')
sns.distplot(accs_knn, hist=False, kde_kws={"shade": True}, label='K-NN')
plt.xlabel('Accuracy')
plt.legend()
plt.savefig('images/model_accuracy_comparison')
plt.show()

# precision
plt.figure(figsize=(12, 6))
plt.grid()
sns.distplot(ps_rfc, hist=False, kde_kws={"shade": True}, label='RFC')
sns.distplot(ps_lr,  hist=False, kde_kws={"shade": True}, label='LR')
sns.distplot(ps_knn, hist=False, kde_kws={"shade": True}, label='K-NN')
plt.xlabel('Precision')
plt.legend()
plt.savefig('images/model_precision_comparison')
plt.show()

# accuracy
plt.figure(figsize=(12, 6))
plt.grid()
sns.distplot(rs_rfc, hist=False, kde_kws={"shade": True}, label='RFC')
sns.distplot(rs_lr,  hist=False, kde_kws={"shade": True}, label='LR')
sns.distplot(rs_knn, hist=False, kde_kws={"shade": True}, label='K-NN')
plt.xlabel('Recall')
plt.legend()
plt.savefig('images/model_recall_precision')
plt.show()




#==============================
# construct an ROC plot
# predict probabilities rather than binary
y_pred_proba_rfc = rfc.predict_proba(X_test)
y_pred_proba_lr  = lr.predict_proba(X_test)
y_pred_proba_knn = knn.predict_proba(X_test)


# get the auc scores
auc_ns      = roc_auc_score(y_test, np.random.randint(0,1,len(y_test)))
auc_rfc     = roc_auc_score(y_test, y_pred_proba_rfc[:, 1])
auc_lr      = roc_auc_score(y_test, y_pred_proba_lr[:, 1])
auc_knn     = roc_auc_score(y_test, y_pred_proba_knn[:, 1])


# get the rates for different thresholds
fpr_ns, tpr_ns, _       = roc_curve(y_test, np.random.randint(0,1,len(y_test)))
fpr_rfc, tpr_rfc, _     = roc_curve(y_test, y_pred_proba_rfc[:, 1])
fpr_lr, tpr_lr, _       = roc_curve(y_test, y_pred_proba_lr[:, 1])
fpr_knn, tpr_knn, _     = roc_curve(y_test, y_pred_proba_knn[:, 1])


#==============================
size = 12
plt.figure(figsize=(size, size))
plt.grid()
plt.plot(fpr_ns,  tpr_ns, linestyle='--', linewidth=size*0.2,
          label='Random model (AUC = {:.2f})'.format(auc_ns))
plt.plot(fpr_rfc,  tpr_rfc,  marker='.', linewidth=size*0.2,
          label='Random Forest (AUC = {:.2f})'.format(auc_rfc))
plt.plot(fpr_lr, tpr_lr, marker='.', linewidth=size*0.2,
          label='Logistic Regression (AUC = {:.2f})'.format(auc_lr))
plt.plot(fpr_knn, tpr_knn, marker='.', linewidth=size*0.2,
          label='K-Nearest Neighbours (AUC = {:.2f})'.format(auc_knn))
plt.xlabel('False Positive Rate', fontsize=size*2)
plt.ylabel('True Positive Rate', fontsize=size*2)
plt.legend(fontsize=size*1.5, loc='lower right')
plt.savefig('images/model_roc_comparison')
plt.show()





#=============================================================================

#=============================================================================
"""
7. Refine the logistic regression model
"""


#============================
# use forward feature selection to refine the model
# will choose features based on the AUC score

# create variables for the features and the best auc score
fts_all = X_train.columns
fts_curr = []
fts_best = None

auc_best = 0

# iteratively add one feature at a time, and compare the auc score
while len(fts_curr) != len(fts_all):
    
    # Get remaining features 
    fts_cand = [f for f in fts_all if f not in fts_curr]
    
    auc_best_this_round = 0
    ft_best_this_round  = None
    
    # compare auc scores achieved through adding one feature at a time to 
    # the set of features
    for ft in fts_cand:
        
        # set the features to test this round
        fts_test = fts_curr + [ft]
        
        # fit the model on the features
        lr = LR().fit(X_train[fts_test], y_train)
        y_pred = lr.predict_proba(X_test[fts_test])
        
        # calculate the auc score
        auc = roc_auc_score(y_test, y_pred[:, 1])
        
        # if auc is better than best auc this round, replace it and the best
        # features
        if auc > auc_best_this_round:
            
            auc_best_this_round = auc
            ft_best_this_round = ft
        
        # if auc is better than the best auc, replace it and the best features
        if auc > auc_best:

            auc_best = auc
            fts_best = fts_test
        
    # Set current features to best features from round
    fts_curr = fts_curr + [ft_best_this_round]
    
    # give status update
    print("\nRound {}: \nSelected feature {} with score {:.3f}"\
          .format(len(fts_curr), 
                  ft_best_this_round, 
                  auc_best_this_round)
          )
    
    # show the current features
    print('Current features are: {}'.format(fts_curr))
    
    # show progress
    print(len(fts_curr), len(fts_all))
            
print("\nBest features were {} ({} total) with score {:.3f}"
      .format(fts_best, 
              len(fts_best), 
              auc_best)
      )




#============================
# fit the model on the best set of features
lr = LR().fit(X_train[fts_best], y_train)
y_pred = lr.predict_proba(X_test[fts_best])

# calculate the final auc
auc = roc_auc_score(y_test, y_pred[:, 1])

# calculate the final scores
y_pred = lr.predict(X_test[fts_best])

accuracy  = acc(y_test, y_pred)
precision = ps(y_test, y_pred)
recall    = rs(y_test, y_pred)


# get the features and their coefficients
ft_strength_df = pd.DataFrame({'Feature':fts_best,
                               'Coefficient':lr.coef_[0]
                               }
                              )

# add features not included in model, but set their coefficient to 0, so that
# they don't affect the model
fts_excluded = list(set(fts_all) ^ set(fts_best))
ft_strength_df = ft_strength_df.append(pd.DataFrame({'Feature':fts_excluded,
                                                     'Coefficient':[0] * len(fts_excluded)
                                                     }
                                                    )
                                       )

ft_strength_df = ft_strength_df.reset_index(drop=True)
ft_strength_df = ft_strength_df.sort_values(by='Coefficient',
                                            ascending=False
                                            )





#============================
plt.figure(figsize=(16, 20))
plt.grid()
sns.barplot(y='Feature', x='Coefficient', 
            data=ft_strength_df,
            dodge=False
            )
plt.yticks(fontsize=15)
plt.ylabel(None)
plt.xlabel('Coefficient', fontsize=17)
plt.xticks(fontsize=15)
plt.tight_layout()
plt.legend(fontsize=20, loc='lower right')
plt.savefig('images/Feature_coefficients')
plt.show()



