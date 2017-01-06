import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sqlalchemy import create_engine
from config import SQLALCHEMY_DATABASE_URI

engine = create_engine(SQLALCHEMY_DATABASE_URI)


def get_db_yr_range(db_name):
    TRAIN_YR_0 = min(pd.read_sql('select distinct year from %s' % db_name, engine).values)[0]
    TRAIN_YR_0 = TRAIN_YR_0+1 #first year has missing data for lagged origin_average and cmd_average
    TRAIN_YR_T = max(pd.read_sql('select distinct year from %s where new_export is not null' % db_name, engine).values)[0]
    TEST_YR = pd.read_sql('select distinct year from %s where new_export is null' % db_name, engine).values[0][0]
    return TRAIN_YR_0, TRAIN_YR_T, TEST_YR

TRAIN_YR_0, TRAIN_YR_T, TEST_YR = get_db_yr_range('mldataset')

features = ['rca', 'export_destination', 'intensity','cmd_average', 'origin_average', 'distance',
             'imports', 'import_origin']

# load train and prediction set
def load_data(dbname, start_yr=TRAIN_YR_0, end_yr=TRAIN_YR_T):
    # to load prediction set set start_yr=TEST_YR and end_yr=TEST_YR
    query = '''
            select * from %s
            where (year >= %d and year <= %d)
            ''' % (dbname, start_yr, end_yr)
    data = pd.read_sql(query, engine).set_index(['origin', 'cmd', 'year'])
    X = data[features]
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    if start_yr != TEST_YR:
        y = data['new_export']
        return X, y
    else:
        return X

# split data for cross-validation
def split_by_yr(X, y, test_range=range(TRAIN_YR_0+1, TRAIN_YR_T+1), leave_last_year_out=True, leave_one_year_out=False):
    if leave_last_year_out:
        for test_year in test_range:
            X_train = X.select(lambda x: x[2] < test_year, axis=0)
            y_train = y.select(lambda x: x[2] < test_year, axis=0)
            X_test = X.select(lambda x: x[2] == test_year, axis=0)
            y_test = y.select(lambda x: x[2] == test_year, axis=0)
            yield test_year, X_train, y_train, X_test, y_test
    if leave_one_year_out:
        for test_year in test_range:
            X_train = X.select(lambda x: x[2] != test_year, axis=0)
            y_train = y.select(lambda x: x[2] != test_year, axis=0)
            X_test = X.select(lambda x: x[2] == test_year, axis=0)
            y_test = y.select(lambda x: x[2] == test_year, axis=0)
            yield test_year, X_train, y_train, X_test, y_test

clfr = XGBClassifier(n_estimators=90, learning_rate=0.1, max_depth=4, min_child_weight=3,
                         seed=42, subsample=0.8)

def cv_by_yr(clfr, X, y, validation_yrs=range(TRAIN_YR_0 + 1, TRAIN_YR_T + 1),
             cv_split='leave_last_year_out', calibrate=True):
    if cv_split=='leave_last_year_out':
            cv_splits = split_by_yr(X, y, validation_yrs, leave_last_year_out=True, leave_one_year_out=False)
    if cv_split=='leave_one_year_out':
            cv_splits = split_by_yr(X, y, validation_yrs, leave_last_year_out=False, leave_one_year_out=True)
    roc_auc_scores = np.array([])
    pr_auc_scores = np.array([])
    log_loss_scores = np.array([])
    for yr, X_train, y_train, X_test, y_test in tqdm(cv_splits):
        clfr = clone(clfr, safe=True)
        if calibrate:
            calibrated_clfr = CalibratedClassifierCV(clfr, cv=3)
            calibrated_clfr.fit(X_train, y_train)
            proba_pred = calibrated_clfr.predict_proba(X_test)[:, 1]
        else:
            clfr.fit(X_train, y_train)
            proba_pred = clfr.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, proba_pred)
        roc_auc_scores = np.append(roc_auc_scores, [roc_auc])
        pr_auc = average_precision_score(y_test, proba_pred)
        pr_auc_scores = np.append(pr_auc_scores, [pr_auc])
        logloss = log_loss(y_test, proba_pred)
        log_loss_scores = np.append(log_loss_scores, [logloss])
        print '\nroc auc: %0.3f, pr auc: %0.3f, log-loss: %0.3f [%d]' % (roc_auc, pr_auc, logloss, yr)
    return validation_yrs, roc_auc_scores, pr_auc_scores, log_loss_scores

def plot_scores(series, labels, years, title, ylim=None):
    plt.figure()
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Score")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid()
    d_range = [datetime(year=yr, month=1, day=1) for yr in years]
    colors =  plt.cm.get_cmap('rainbow')(np.linspace(0, 1, len(series)))
    for series, lab, color in zip(series, labels, colors):
        plt.plot(d_range, series, 'o-', color=color, label=lab)
    plt.legend(loc="best")
    plt.savefig('docs/images/roc_curve_no_labels.png' % title)

def roc_curve(clfr, X, y, test_year= TRAIN_YR_T, calibrate=True):
    cv = split_by_yr(X, y, [test_year])
    _, X_train, y_train, X_test, y_test = next(cv)
    if calibrate:
        clfr = CalibratedClassifierCV(clfr, cv=3)
    clfr.fit(X_train, y_train)
    proba_pred = clfr.predict_proba(X_test)[:, 1]
    fpr, tpr, threholds = metrics.roc_curve(y_test.as_matrix(), proba_pred)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.grid()
    plt.title('ROC curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig('docs/images/roc_curve_no_labels.png')
    #add labels
    threholds = pd.DataFrame(data=np.array([fpr,tpr,threholds]).transpose(), columns=['fpr','tpr','threshold'])
    for threshold_value in [0.03, 0.05,0.07, 0.1, 0.15, 0.2, 0.25]:
        label = 'prob. threshold = %0.2f' % threshold_value
        label_ypos = threholds[threholds.threshold<threshold_value].tpr.values[0]
        label_xpos = threholds[threholds.threshold<threshold_value].fpr.values[0]
        plt.annotate(label, xy=(label_xpos, label_ypos), xytext=(150, -40),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow'),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    plt.savefig('docs/images/roc_curve.png')


def print_feature_importance(clfr, X, y):
    clfr.fit(X,y)
    for importance, feature in sorted(zip(clfr.feature_importances_, X.columns), reverse=True):
        print "%0.3f - %s" % (importance, feature)

def plot_learning_curve(clfr, title, X, y, ylim=None,
                        n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        clfr, X, y, cv=3, scoring='roc_auc', n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training ROC AUC score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation ROC AUC score")

    plt.legend(loc="best")
    plt.savefig('docs/images/learning_curve.png')
    return plt


def predictions(clfr, X, y, prediction_set, calibrate=True, drop_threshold=0.0):
    if calibrate:
        clfr = CalibratedClassifierCV(clfr, cv=3)
    clfr.fit(X, y)
    proba_pred = clfr.predict_proba(prediction_set)[:, 1]
    y_pred = pd.DataFrame(proba_pred, columns=['proba_pred'], index=prediction_set.index).reset_index()
    ctry_names = pd.read_sql('select code, name as ctry_name from countries', engine)
    y_pred = y_pred.merge(ctry_names, how='left', left_on='origin', right_on='code')
    y_pred = y_pred.drop('code', axis=1)
    cmd_names = pd.read_sql('select code, name as cmd_name from commodities', engine)
    y_pred = y_pred.merge(cmd_names, how='left', left_on='cmd', right_on='code')
    y_pred = y_pred.drop('code', axis=1)
    y_pred = y_pred.drop('year', axis=1)
    y_pred = y_pred[y_pred.proba_pred>drop_threshold]
    return y_pred


# get cmd-parent names to merge with predictions - used for web visualisations
def parent_names(year = TRAIN_YR_T):
    # returns cmd parent names
    query = '''
                select code as cmd, name_2dg, name_1dg
                from commodities
                ''' % year
    return pd.read_sql(query, engine)


def predictions_to_csv(predictions, parent_names, target):
    predictions = predictions.merge(parent_names, how='left', on='cmd')

    # edit country codes for compatibility with topojson
    predictions.origin = predictions.origin.apply(lambda x: str(x).zfill(3))
    ctry_map_dict = {code: code for code in list(set(predictions.origin.values))}
    ctry_map_dict['842'] = '840'  # USA
    ctry_map_dict['699'] = '356'  # India
    ctry_map_dict['381'] = '380'  # Italy
    ctry_map_dict['251'] = '250'  # France
    ctry_map_dict['579'] = '578'  # Norway
    ctry_map_dict['757'] = '756'  # Switzerland
    predictions.origin = predictions.origin.apply(lambda x: ctry_map_dict[x])

    predictions['target'] = target
    try:
        current_file = pd.read_csv('docs/predicitons.csv', sep='\t', encoding='utf-8')
        current_file = current_file[current_file.target != target]
        predictions = pd.concat([current_file, predictions], axis=0)
    except Exception:
        pass
    predictions.to_csv('docs/predicitons.csv', index=None, sep='\t', encoding='utf-8')



if __name__ == "__main__":

    X, y = load_data('mldataset')

    years, roc_auc_scores, pr_auc_scores, log_loss_scores = cv_by_yr(clfr, X, y,
                                                            cv_split='leave_last_year_out', calibrate=True)

    plot_scores([roc_auc_scores, log_loss_scores], ['ROC AUC', 'Log loss'], years,
                'Leave-last-year-out cross validation', ylim=(0,1))

    roc_curve(clfr, X, y)


    # predictions using mldataset
    prediction_set = load_data('mldataset', start_yr=TEST_YR, end_yr=TEST_YR)
    predictions = predictions(clfr, X, y, prediction_set, calibrate=True)
    parent_names = parent_names()
    predictions_to_csv(predictions, parent_names, 1)

    # predictions using mldataset2
    TRAIN_YR_0, TRAIN_YR_T, TEST_YR = get_db_yr_range('mldataset2')
    X, y = load_data('mldataset2')
    prediction_set = load_data('mldataset2', start_yr=TEST_YR, end_yr=TEST_YR)
    predictions = predictions(clfr, X, y, prediction_set, calibrate=True, drop_threshold=0.005)
    predictions_to_csv(predictions, parent_names, 2)



