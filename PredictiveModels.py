from sys import exit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

plt.rcParams['font.size'] = 14

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN


def load_data(fname, path='data/'):
    with open('%s%s' % (path, fname), mode='rU') as f:
        iter_csv = pd.read_csv(f, sep=',', header=0, iterator=True, chunksize=50000)
        df = pd.concat(iter_csv, ignore_index=True)
    df.index = [df.cmd,df.origin]
    print 'data loaded'
    return df

def lda(features, y):
    lda = LinearDiscriminantAnalysis()
    lda.fit(features, y)
    component = pd.DataFrame(data=lda.transform(features),index=features.index)
    return component

def get_first_k(list, k):
    # sorts list of strings based on numeric characters within each string, returns first k elements
    sorted_list = sorted(list, key = lambda x: int(re.sub("[^0-9]","",x)))
    return sorted_list[:k]

def preprocess(df, target, ctry_dummies=False, cmd_dummies=False, LDA=False, k=20, logistic_coeff=None):

    if logistic_coeff is not None:
        ctry_logistic_coeff = logistic_coeff[0].to_frame()
        ctry_logistic_coeff.columns = ['ctry_logistic_coeff']
        df = df.merge(ctry_logistic_coeff, how='left', left_on='origin',right_index=True)
        cmd_logistic_coeff = logistic_coeff[1].to_frame()
        cmd_logistic_coeff.columns = ['cmd_logistic_coeff']
        df = df.merge(cmd_logistic_coeff, how='left', left_on='cmd',right_index=True)
        df.fillna(value=0,inplace=True)

    y = df[target]
    df = df.loc[:, (df != 0).any(axis=0)] # delete features that are all zero

    rca_neighbours = get_first_k([col for col in df.columns if 'neighbour' in col and 'rca' in col], k)
    destination_neighbours = get_first_k([col for col in df.columns if 'neighbour' in col and 'dest' in col], k)
    intensity_neighbours = get_first_k([col for col in df.columns if 'neighbour' in col and 'intensity' in col], k)
    logistic_coeff = [col for col in df.columns if 'logistic_coeff' in col]

    if LDA:
        rca_neighbours = lda(df[rca_neighbours], y)
        destination_neighbours = lda(df[destination_neighbours], y)
        intensity_neighbours = lda(df[intensity_neighbours], y)
        X = pd.concat([rca_neighbours,destination_neighbours,intensity_neighbours,df[logistic_coeff]], axis=1)
        features = ['rca_neighbours','destination_neighbours','intensity_neighbours']+logistic_coeff

    else:
        features = rca_neighbours + destination_neighbours + intensity_neighbours + logistic_coeff
        X = df[features]

    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=features, index=df.index)

    if ctry_dummies:
        origin_dummies = pd.get_dummies(df.origin, prefix='origin_dummy', drop_first=True)
        X = pd.concat([X, origin_dummies], axis=1)
    if cmd_dummies:
        cmd_dummies = pd.get_dummies(df.cmd, prefix='cmd_dummy', drop_first=True)
        X = pd.concat([X, cmd_dummies], axis=1)
    return X, y

def logistic_dummy_coeff(X_train, y_train):
    dummies = [col for col in X_train.columns if 'dummy' in col]
    X_train = X_train[dummies]
    X_train.columns = [int(re.sub("[^0-9]","",col)) for col in X_train.columns]
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return pd.Series(clf.coef_[0], index= X_train.columns)


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
    y_pred = clf.predict(X_test)
    if y_pred == None:
        y_pred = clf.predict(X_test)
    y_hat = clf.predict(X_train)
    print clf, '\n\n'
    print 'in-sample accuracy: ', metrics.accuracy_score(y_train, y_hat)
    print 'out-sample accuracy: ', metrics.accuracy_score(y_test, y_pred)
    print 'baseline: ', metrics.accuracy_score(y_test, np.zeros_like(y_test))
    try:
        y_pred_prob = clf.predict_proba(X_test)[:, 1]
    except:
        y_pred_prob = clf.decision_function(X_test)
    print 'auc: ', metrics.roc_auc_score(y_test, y_pred_prob)
    confusion_matrix(clf, X_train, X_test, y_train, y_test)

def grid_search(param_grid, clf, metric, X, y):
    gs = GridSearchCV(clf,param_grid,cv=10,scoring=metric)
    gs.fit(X, y)
    print "Best parameter set based on %s:" % metric
    print gs.best_params_
    print metric,':', gs.best_score_

param_grid = {
    # 'max_features': range(25, 35, 2),
    # 'max_depth': range(7, 11, 1)
    'scale_pos_weight': np.arange(0.1, 1.1, 0.2)
    # 'bootstrap': [True, False]
    # 'penalty': ['l2','l1'],
    # 'loss' : ['squared_hinge','hinge'],
    # 'dual' : [True],
    # 'C' : np.arange(0.01,1.01,0.2),
    # 'class_weight' : [None,'balanced']
}

def roc_curve(clf, X_train, X_test, y_train, y_test):
    # roc_auc: probability the model will rank a randomly chosen positive instance higher than a randomly chosen negative one
    clf.fit(X_train, y_train)
    print_scores(X_train, X_test, y_train, y_test, clf)
    try:
        y_pred_prob = clf.predict_proba(X_test)[:, 1]
    except:
        y_pred_prob = clf.decision_function(X_test)
    fpr, tpr, threholds = metrics.roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()
    return pd.DataFrame(data=np.array([fpr,tpr,threholds]).transpose())

def cv(clf,X,y,metric):
    scores = cross_val_score(clf, X, y, cv=10, scoring=metric)
    print '%s \n mean %s score: %f' % (clf, metric, scores.mean())
    return scores.mean()

def print_cv_score(classifiers, X, y):
    for metric in ['accuracy', 'roc_auc']:
        for clf in classifiers:
            cv(clf, X, y, metric)

def predict(clf, X_train, X_test, y_train, y_test, threshold = 0.5):
    # alternative thresholds to trade-off accuracy/tpr/fpr
    clf.fit(X_train, y_train)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = np.array([prob > threshold for prob in y_pred_prob])
    print_scores(X_train, X_test, y_train, y_test, clf, y_pred)
    return y_pred

def get_dummy_coef(df):
    X, y = preprocess(df,'has_export', ctry_dummies=True, cmd_dummies=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    ctry_dummy_coef = logistic_dummy_coeff(X_train, y_train)

    X, y = preprocess(df,'has_export', ctry_dummies=False, cmd_dummies=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    cmd_dummy_coef = logistic_dummy_coeff(X_train, y_train)
    return ctry_dummy_coef, cmd_dummy_coef


def get_scores_by_k(df, clf, k_range, metric='accuracy'):
    # use to choose k: number of neighbours to use in LDA
    ctry_dummy_coef, cmd_dummy_coef = get_dummy_coef(df)
    scores = []
    for k in k_range:
        X, y = preprocess(df, 'has_export', ctry_dummies=False, cmd_dummies=False, LDA=True, k=k,
                          logistic_coeff=(ctry_dummy_coef, cmd_dummy_coef))
        score = cv(clf, X, y, metric)
        scores.append(score)
    print '%s scores:' % metric
    print zip(k_range, scores)
    x, y = zip(*zip(k_range, scores))
    plt.scatter(x, y)
    plt.show()


def resample(X, y, k=14, m=12):
    sm = SMOTEENN(smote=SMOTE(kind='svm', k_neighbors=k, m_neighbors=m), random_state=42)
    sm.fit_sample(X, y)
    X_resampled, y_resampled = sm.sample(X, y)
    X_resampled = pd.DataFrame(data=X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled)
    return X_resampled, y_resampled

def build_ensemble(X, y, classifiers):
    ensemble = pd.DataFrame()
    for clf in classifiers:
        proba_cv = cross_val_predict(clf, X, y, cv=10, method='predict_proba')[:, 1]
        proba_cv = pd.Series(data=proba_cv)
        ensemble = pd.concat([ensemble, proba_cv], axis=1)
        ensemble.columns = ['clf_%d' % (i+1) for i in range(ensemble.shape[1])]
    return ensemble

def ensemble_score(ensemble, y, metric):
    blender = XGBClassifier()
    score = cv(blender, ensemble, y, metric)
    return score

def resample_tuner(X, y, k_range, m_range, metric='accuracy'):
    # only one of k_range, m_range should have length > 1
    scores = []
    for k in k_range:
        for m in m_range:
            X_resampled, y_resampled = resample(X, y, k=k, m=m)
            ensemble_resampled = build_ensemble(X_resampled, y_resampled, classifiers)
            score = ensemble_score(ensemble_resampled, y_resampled, metric)
            scores.append(score)
        print zip(m_range, scores)
        x, y = zip(*zip(m_range, scores))
        plt.scatter(x, y)
        plt.show()

def pred_proba(X, y, classifiers):
    X_resampled, y_resampled = resample(X, y, k=14, m=12)
    ensemble_resampled = build_ensemble(X_resampled, y_resampled, classifiers)
    ensemble = build_ensemble(X, y, classifiers) # ??
    blender = XGBClassifier()
    blender.fit(ensemble_resampled, y_resampled)
    pred_proba = blender.predict_proba(ensemble)[:, 1] # ???
    pred_proba = pd.Series(pred_proba, index=y.index)
    pred_proba = pd.concat([y, pred_proba], axis=1)
    pred_proba.columns = ['has_export', 'pred_prob']
    return pred_proba

if __name__ == "__main__":

    try:
        df = load_data('mldataset_4dg.csv')
    except IOError:
        df = load_data('sample_mldataset_4dg.csv', path='')

    ctry_dummy_coef, cmd_dummy_coef = get_dummy_coef(df)
    X, y = preprocess(df,'has_export', ctry_dummies=False, cmd_dummies=False, LDA=True, k=200, logistic_coeff=(ctry_dummy_coef,cmd_dummy_coef))

    classifiers = [
        XGBClassifier(),                     # cv accuracy = 0.825(0.759) cv auc = 0.829(0.845)   unbalanced(resampled)
        RandomForestClassifier(),            # cv accuracy = 0.812(0.833) cv auc = 0.780(0.911)   unbalanced(resampled)
        LogisticRegression()                 # cv accuracy = 0.839(0.792) cv auc = 0.850(0.877)   unbalanced(resampled)
        #Ensemble (XGBClassifier)            # cv accuracy = 0.838(0.856) cv auc = 0.849(0.928)   unbalanced(resampled)
    ]

    print_cv_score(classifiers, X, y)

    X_resampled, y_resampled = resample(X, y, k=14, m=12)
    print_cv_score(classifiers, X_resampled, y_resampled)

    ensemble_resampled = build_ensemble(X_resampled, y_resampled, classifiers)
    ensemble = build_ensemble(X, y, classifiers)

    print ensemble_resampled.groupby(y_resampled).mean()
    print ensemble.groupby(y.values).mean()

    print ensemble_score(ensemble, y, 'accuracy')
    print ensemble_score(ensemble, y, 'roc_auc')
    print ensemble_score(ensemble_resampled, y_resampled, 'accuracy')
    print ensemble_score(ensemble_resampled, y_resampled, 'roc_auc')

    pred_proba = pred_proba(X, y, classifiers)
    pred_proba['pred_prob'].hist(by=pred_proba['has_export'])
    plt.show()




