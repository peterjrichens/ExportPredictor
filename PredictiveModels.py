from sys import exit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import xgboost


def load_data(fname, path='data/'):
    with open('%s%s' % (path, fname), mode='rU') as f:
        iter_csv = pd.read_csv(f, sep=',', header=0, iterator=True, chunksize=50000)
        df = pd.concat(iter_csv, ignore_index=True)
    df.index = [df.cmd,df.origin]
    print 'data loaded'
    return df

def components(n, X, y, plot_matrix=False):
    pca = PCA(n_components=n)
    components = pca.fit_transform(X)
    components = pd.DataFrame(components, index=X.index)
    if plot_matrix:
        pd.scatter_matrix(pd.concat([y,components],axis=1))
        plt.show()
    return components

def preprocess(df, target, ctry_dummies=False, cmd_dummies=False, pca=False, numb_components=4):

    y = df[target]
    df = df.loc[:, (df != 0).any(axis=0)] # delete features that are all zero

    rca_similarity = [col for col in df.columns if 'similarity' in col and 'rca' in col]
    destination_similarity = [col for col in df.columns if 'similarity' in col and 'dest' in col]
    intensity_similarity = [col for col in df.columns if 'similarity' in col and 'intensity' in col]

    features = rca_similarity + destination_similarity + intensity_similarity
    X = df[features]
    X = pd.DataFrame(StandardScaler(with_mean=False).fit_transform(X), columns=features, index=df.index) #doesn't destroy sparsity

    nb_zeros = (X == 0).astype(int).sum(axis=1).sum()
    total_size =  X.shape[0] * X.shape[1]
    print 'sparsity in feature matrix:',float(nb_zeros)/total_size

    if pca:
        components(numb_components, X, y, plot_matrix=False)

    if ctry_dummies:
        origin_dummies = pd.get_dummies(df.origin, prefix='origin_dummy', drop_first=True)
        X = pd.concat([X, origin_dummies], axis=1)
    if cmd_dummies:
        cmd_dummies = pd.get_dummies(df.cmd, prefix='cmd_dummy', drop_first=True)
        X = pd.concat([X, cmd_dummies], axis=1)

    return X, y

def error_at_ctry_level(classifiers,X,y,cmds,ctries):
    # calculates number of new exports for each country and returns sum of square errors
    for clf in classifiers:
        clf.fit(X,y)
        prediction = pd.Series(clf.predict(X), index=[cmds,ctries])
        num_new_exports_by_ctry = pd.concat([cmds,ctries,y,prediction],axis=1).groupby(ctries).sum()
        num_new_exports_by_ctry.columns = ['cmd','origin','has_export','prediction']
        sse = sum((num_new_exports_by_ctry.has_export - num_new_exports_by_ctry.prediction) ** 2)
        print clf,'\n\nsum of errors at country-level: ',sse,'\n'


def confusion_matrix(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print clf, '\n\n'
    print '\nConfusion matrix:\n', confusion_matrix,'\n'

    print "When it's actually yes, how often does it predict yes?"
    print 'true positive rate/sensitivity/recall:', metrics.recall_score(y_test, y_pred),'\n'

    print 'When it predicts yes, how often is it correct?'
    print 'precision: ', float(confusion_matrix[1, 1]) / (confusion_matrix[0, 1] + confusion_matrix[1, 1]),'\n'

    print "When it's actually no, how often does it predict yes?"
    print 'false positive rate: ', float(confusion_matrix[0, 1]) / (confusion_matrix[0, 0] + confusion_matrix[0, 1])

def print_scores(X_train, X_test, y_train, y_test, clf, y_pred = None):
    if y_pred == None:
        y_pred = clf.predict(X_test)
    y_hat = clf.predict(X_train)
    print clf, '\n\n'
    print 'in-sample accuracy: ', metrics.accuracy_score(y_train, y_hat)
    print 'out-sample accuracy: ', metrics.accuracy_score(y_test, y_pred)
    print 'baseline: ', metrics.accuracy_score(y_test, np.zeros_like(y_test))
    print 'auc: ', metrics.roc_auc_score(y_test, y_pred)
    confusion_matrix(clf, X_train, X_test, y_train, y_test)

def grid_search(param_grid, clf, metric, X_train, X_test, y_train, y_test):
    gs = GridSearchCV(clf,param_grid,cv=10,scoring=metric)
    gs.fit(X_train, y_train)
    print clf, '\n\n'
    print "Best parameters set found on training set:"
    print gs.best_params_
    print_scores(X_train, X_test, y_train, y_test, gs)


def roc_curve(clf, X_train, X_test, y_train, y_test):
    # roc_auc: probability the model will rank a randomly chosen positive instance higher than a randomly chosen negative one
    clf.fit(X_train, y_train)
    print_scores(X_train, X_test, y_train, y_test, clf)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, threholds = metrics.roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()
    return pd.DataFrame(data=np.array([fpr,tpr,threholds]).transpose())

def cv(clf,X,y,metric):
    scores = cross_val_score(clf, X, y, cv=10, scoring=metric)
    print '%s \n mean %s score: %f' % (clf, metric, scores.mean())

def predict(clf, X_train, X_test, y_train, y_test, threshold = 0.5):
    # alternative thresholds to trade-off accuracy/tpr/fpr
    clf.fit(X_train, y_train)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = np.array([prob > threshold for prob in y_pred_prob])
    print_scores(X_train, X_test, y_train, y_test, clf, y_pred)
    return y_pred


if __name__ == "__main__":

    try:
        df = load_data('mldataset_4dg.csv')
    except IOError:
        df = load_data('sample_mldataset_4dg.csv', path='')

    X, y = preprocess(df,'has_export', ctry_dummies=False, cmd_dummies=False, pca=False)


    classifiers = [
        xgboost.XGBClassifier(scale_pos_weight=3.18),  # accuracy =  0.754 auc = 0.737 TODO: gridsearch
        RandomForestClassifier(max_features=33, max_depth=10),  # accuracy = 0.81 auc =  0.605
        LogisticRegression(penalty='l1')  # (with dummies) accuracy = 0.838 auc =  0.69
    ]

    error_at_ctry_level(classifiers, X, y, df.cmd, df.origin)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=88)

    for clf in classifiers:
        print_scores(X_train, X_test, y_train, y_test, clf)
        cv(clf, X, y, 'accuracy')
        cv(clf, X, y, 'roc_auc')

    param_grid = {
        'max_features': range(25, 35, 2),
        'max_depth': range(10, 20, 2)
    }

    grid_search(param_grid, classifiers[1], 'roc_auc', X_train, X_test, y_train, y_test)
