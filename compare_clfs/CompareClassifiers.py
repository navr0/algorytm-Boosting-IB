import os

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone

from classifier import BoostingIB
from classifier.BoostingIB import Boosting_IB
from classifier.OLA import OLA
from tests.TStudentTest import TStudentTest


class CompareClassifiers:

    def __init__(self):
        self.compare_path = f'result_directory/compare.csv'
        self.pool_clfs = [
            Boosting_IB(repeats=5),
            DecisionTreeClassifier(),
            GaussianNB()
        ]

        self.base_clfs = [
            Boosting_IB(repeats=5),
            GaussianNB(),
            AdaBoostClassifier(),
            DecisionTreeClassifier(),
            OLA(pool_classifiers=self.pool_clfs)
        ]


    def compare(self):
        file = open(self.compare_path, 'w')
        file.write(f'Dataset;BoostingIB;GaussianNB;AdaBoost;DecisionTree;OLA')
        dir_path = "datasets"
        csv_files = os.listdir(dir_path)


        for csv_idx, filename in enumerate(csv_files):
            file_path = f'{dir_path}/{filename}'
            current_data = np.loadtxt(file_path, delimiter=',')
            self.file.write(f'\n{filename};')

            x = current_data[:, :-1]
            y = current_data[:, -1]

            rskf = RepeatedStratifiedKFold(n_repeats=2, n_splits=5)
            print(filename)
            for idx_rskf, (idx_train, idx_test) in enumerate(rskf.split(x, y)):
                x_train, y_train = x[idx_train], y[idx_train]
                x_test, y_test = x[idx_test], y[idx_test]
            csv_scores = []
            for index_clf, clf in enumerate(self.base_clfs):
                clf.fit(x_train, y_train)
                clf_result = clf.predict(x_test)
                score = float('%.3f' % balanced_accuracy_score(y_true=y_test, y_pred=clf_result))
                std = float('%.3f' % np.std(score))
                self.file.write(f'{score}({std});')


            save_name = filename.split('.')[0]
            numpy_results = f'tests/numpy_files/{save_name}_results.npy'
            test_scores = []
            for i in range(len(self.base_clfs)):
                test_scores.append(csv_scores)
            np.save(numpy_results, test_scores)
            file.write('\n')



        return ""
