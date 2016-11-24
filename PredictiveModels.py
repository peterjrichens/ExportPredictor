from sys import exit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
import seaborn as sns

from patsy import dmatrices, dmatrix, standardize
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
import xgboost


with open('%s/%s' % ('data', 'mldataset_4dg.csv'), mode='rU') as f:
    iter_csv = pd.read_csv(f, sep=',', header=0, iterator=True, chunksize=50000)
    df = pd.concat(iter_csv, ignore_index=True)
print 'data loaded'
df.index = [df.cmd,df.origin]


origin_dummies = pd.get_dummies(df.origin, prefix='origin_dummy', drop_first=True)
cmd_dummies = pd.get_dummies(df.cmd, prefix='cmd_dummy', drop_first=True)
df = pd.concat([df, origin_dummies, cmd_dummies], axis=1)

#df.origin = df.origin.apply(str) # Patsy treats strings as categorical variables
#df.cmd = df.cmd.apply(str)
#y, X = dmatrices('has_export ~ origin + cmd + origin:standardize(rca_proximity) + origin:standardize(destination_proximity) + origin:standardize(intensity_proximity)', df)

y = df['has_export']
df = df.loc[:, (df != 0).any(axis=0)] # delete features that are all zero

rca_similarity = [col for col in df.columns if 'similarity' in col and 'rca' in col] #rf0.798
destination_similarity = [col for col in df.columns if 'similarity' in col and 'dest' in col] #rf0.787
intensity_similarity = [col for col in df.columns if 'similarity' in col and 'intensity' in col] # rf0.807
all = [col for col in df.columns if 'similarity' in col] #rf0.811

dummies = [col for col in df.columns if 'dummy' in col]

features = rca_similarity+destination_similarity+intensity_similarity

X = df[features]

X = pd.DataFrame(StandardScaler(with_mean=False).fit_transform(X), columns=features, index=df.index) #doesn't destroy sparsity

nb_zeros = (X == 0).astype(int).sum(axis=1).sum()
total_size =  X.shape[0] * X.shape[1]
print 'sparsity ratio: ',float(nb_zeros)/total_size


def components(n, X):
    pca = PCA(n_components=n)
    components = pca.fit_transform(X)
    components = pd.DataFrame(components, index=X.index)
    return components

#df2 = pd.concat([y,components(2, X)],axis=1)

#print df2.groupby(y).describe()
#pd.scatter_matrix(df2)
#plt.show()
#exit()

#X = pd.concat([X,df[dummies]],axis=1) # improve logistic, rf better without
#print X.describe()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=88)
'''
param_grid = {
    'max_depth': range(25,35,2),
    'max_features': range(10,20,2)
}
clf = GridSearchCV(RandomForestClassifier(),param_grid,cv=10,scoring='accuracy')
clf.fit(X_train,y_train)
print "Best parameters set found on development set:"
print clf.best_params_

print "Detailed classification report:"
y_true, y_pred = y_test, clf.predict(X_test)
y_hat = clf.predict(X_train)
print 'in-sample accuracy: ', metrics.accuracy_score(y_train, y_hat)
print 'out-sample accuracy: ', metrics.accuracy_score(y_test, y_pred)
print 'baseline: ', metrics.accuracy_score(y_test,np.zeros_like(y_test))
print 'auc: ', metrics.roc_auc_score(y_test, y_pred)
exit()
'''

model = xgboost.XGBClassifier(scale_pos_weight=3.18) # accuracy =  0.754 auc = 0.737 TODO: gridsearch
#model = LinearSVC(penalty='l1', dual=False, random_state=1) # accuracy = 0.7825 auc = 0.528
#model = RandomForestClassifier(max_features=33,max_depth=10) # accuracy = 0.81 auc =  0.605
#model = LogisticRegression(penalty='l1') # (with dummies) accuracy = 0.838 auc =  0.69

model.fit(X,y)
y_hat = model.predict(X)
df['y_hat'] = pd.Series(model.predict(X), index=[df.cmd,df.origin])
print df[['has_export','y_hat']].groupby('origin').sum()

exit()

model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
#y_pred = np.array([prob > 0.5 for prob in y_pred_prob])
y_hat = model.predict(X_train)

print 'in-sample accuracy: ', metrics.accuracy_score(y_train, y_hat)
print 'out-sample accuracy: ', metrics.accuracy_score(y_test, y_pred)
print 'baseline: ', metrics.accuracy_score(y_test,np.zeros_like(y_test))

fpr, tpr, threholds = metrics.roc_curve(y_test, y_pred_prob)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print '\nConfusion matrix:\n',confusion_matrix

print 'true positive rate:' , metrics.recall_score(y_test, y_pred) # When it's actually yes, how often does it predict yes?
print 'precision: ', float(confusion_matrix[1,1])/(confusion_matrix[0,1]+confusion_matrix[1,1]) # When it predicts yes, how often is it correct?
print 'false positive rate: ', float(confusion_matrix[0,1])/(confusion_matrix[0,0]+confusion_matrix[0,1]) # When it's actually no, how often does it predict yes?
print 'auc: ', metrics.roc_auc_score(y_test, y_pred)
# roc_auc: probability the model will rank a randomly chosen positive instance higher than a randomly chosen negative one
#exit()
#roc = pd.DataFrame(data=np.array([fpr,tpr,threholds]).transpose())
#print roc.tail()
plt.plot(fpr, tpr, label='model')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
# TPR is also known as sensitivity/recall, and FPR is one minus the specificity or true negative rate
plt.show()

#exit()

def cv(model,features):


    X = df[features]

    scores = cross_val_score(model, X, y, cv=10, scoring='roc_auc')

    print '%s \n features: %r\n lags=%r \n mean score: %f' % (model, features, scores.mean())



