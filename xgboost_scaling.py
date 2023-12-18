import pandas as pd
import numpy
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore')

pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.max_colwidth', None)

PATH = '/home/steven/PythonProjects/XGBoost_Scaling/Diabetes_Balanced_Binary.csv'
RANDOM_STATE = 42

# ################################################################

def train_and_fit(clf, x_data, y_data):

    scores, mean_f1 = 0.0, 0.0

    X_train, X_cv, y_train, y_cv = train_test_split(x_data, y_data, test_size=.20, random_state=RANDOM_STATE)
    fit_data = clf.fit(X_train, y_train)
    preds = clf.predict(X_cv)

    # Calculate performance metrics
    scores += metrics.accuracy_score(y_cv, preds)
    prec, rec, _, _ = metrics.precision_recall_fscore_support(y_cv, preds, pos_label=1,
                                                                average='binary')
    
    mean_f1 += 2 * numpy.mean(prec) * numpy.mean(rec) / (numpy.mean(prec) + numpy.mean(rec))

    # Calculate mean performance metrics
    mean_accuracy = scores

    f1 = mean_f1

    metrics_dict = {'Accuracy': float(mean_accuracy),
                    'F1-Score': float(f1)}
    
    return (metrics_dict)


# ################################################################

# Calling the functions

data = pd.read_csv(str(PATH))

x_data = data.iloc[:, 1:]
y_data = data.iloc[:, 0]

classifier_list = [GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=1),
                   GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000, max_depth=1),
                   GradientBoostingClassifier(learning_rate=0.001, n_estimators=10000, max_depth=1),
                   GradientBoostingClassifier(learning_rate=0.0001, n_estimators=100000, max_depth=1),
                   GradientBoostingClassifier(learning_rate=0.001, n_estimators=100, max_depth=1),
                   GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000, max_depth=1),
                   GradientBoostingClassifier(learning_rate=0.1, n_estimators=10000, max_depth=1),
                   GradientBoostingClassifier(learning_rate=1.0, n_estimators=100000, max_depth=1)]

for classifier in classifier_list:
    model_metrics = train_and_fit(classifier, x_data, y_data)
    print('Gradient Boosting: ', model_metrics, str(classifier).split('(')[0], (str(classifier).split('(')[1]).replace('\n', ' '))

classifier_list = [AdaBoostClassifier(learning_rate=0.1, n_estimators=100),
                   AdaBoostClassifier(learning_rate=0.001, n_estimators=1000),
                   AdaBoostClassifier(learning_rate=0.001, n_estimators=10000),
                   AdaBoostClassifier(learning_rate=0.0001, n_estimators=100000),
                   AdaBoostClassifier(learning_rate=0.001, n_estimators=100),
                   AdaBoostClassifier(learning_rate=0.01, n_estimators=1000),
                   AdaBoostClassifier(learning_rate=0.1, n_estimators=10000),
                   AdaBoostClassifier(learning_rate=1.0, n_estimators=100000)]

for classifier in classifier_list:
    model_metrics = train_and_fit(classifier, x_data, y_data)
    print('AdaBoost: ', model_metrics, str(classifier).split('(')[0], (str(classifier).split('(')[1]).replace('\n', ' '))


classifier_list = [XGBClassifier(learning_rate=0.1, n_estimators=100),
                   XGBClassifier(learning_rate=0.001, n_estimators=1000),
                   XGBClassifier(learning_rate=0.001, n_estimators=10000),
                   XGBClassifier(learning_rate=0.0001, n_estimators=100000),
                   XGBClassifier(learning_rate=0.001, n_estimators=100),
                   XGBClassifier(learning_rate=0.01, n_estimators=1000),
                   XGBClassifier(learning_rate=0.1, n_estimators=10000),
                   XGBClassifier(learning_rate=1.0, n_estimators=100000)]

for classifier in classifier_list:
    model_metrics = train_and_fit(classifier, x_data, y_data)
    print('XGBoost: ', model_metrics, str(classifier).split('(')[0], (str(classifier).split('(')[1]).replace('\n', ' '))
