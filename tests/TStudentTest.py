import os

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from classifier.BoostingIB import Boosting_IB
from classifier.OLA import OLA

from scipy import stats


class TStudentTest:
    def __init__(self):
        self.pool_clfs = [
            KNeighborsClassifier(),
            DecisionTreeClassifier(),
            GaussianNB()
        ]
        self.clfs = [
            Boosting_IB(repeats=5),
            GaussianNB(),
            AdaBoostClassifier(),
            DecisionTreeClassifier(),
            OLA(pool_classifiers=self.pool_clfs)
        ]
        self.clf_names = [
            'BoostingIB',
            'GaussianNB',
            'AdaBoost',
            'DecisionTree',
            'OLA'
        ]
        self.dir_path = 'datasets'
        self.compare_path = f'result_directory/compare.csv'

    def test(self):
        csv_files = os.listdir(self.dir_path)
        result = np.zeros((len(csv_files), 10, len(self.clfs)))

        print("test poczatek")

        csv_compare = open(self.compare_path, 'w')
        csv_compare.write('DATASETS;')
        for clf_name in self.clf_names:
            csv_compare.write(f'{clf_name};')
        csv_compare.write('\n')

        for csv_idx, filename in enumerate(csv_files):
            filename_without_csv = filename.split('.')[0]
            csv_compare.write(f'\n{filename_without_csv};')
            file_path = f'{self.dir_path}/{filename}'
            current_data = np.loadtxt(file_path, delimiter=',')

            print(f'test csv {csv_idx+1}')

            x = current_data[:, :-1]
            y = current_data[:, -1]

            rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2)
            std_scores = [[0 for i in range(10)] for j in range(len(self.clfs))]
            for rskf_index, (train_index, test_index) in enumerate(rskf.split(x, y)):
                x_train, y_train = x[train_index], y[train_index]
                x_test, y_test = x[test_index], y[test_index]

                for clf_idx, clf in enumerate(self.clfs):
                    clf.fit(x_train, y_train)
                    clf_predict = clf.predict(x_test)
                    score = balanced_accuracy_score(y_true=y_test, y_pred=clf_predict)
                    result[csv_idx, rskf_index, clf_idx] = score
                    std_scores[clf_idx][rskf_index] = score

            for clf_index, clf in enumerate(self.clfs):
                mean = float( '%.3f' % np.mean(std_scores[clf_index][:]))
                std = float( '%.2f' % np.std(std_scores[clf_index][:]))
                csv_compare.write(f'{mean}({std});')

        final_file = "results.npy"
        np.save(final_file, result)

        alpha = 0.05
        datasets = np.load(f'results.npy')

        for csv_idx, filename in enumerate(csv_files):
            data = datasets[csv_idx]

            t_student_matrix = np.zeros((len(self.clfs), len(self.clfs)))
            p_matrix = np.zeros((len(self.clfs), len(self.clfs)))
            tf_matrix = np.zeros((len(self.clfs), len(self.clfs)), dtype=bool)
            alpha_matrix = np.zeros((len(self.clfs), len(self.clfs)), dtype=bool)
            advantage_matrix = np.zeros((len(self.clfs), len(self.clfs)), dtype=bool)

            print("test")

            for idx_first_clf, first_clf in enumerate(self.clfs):
                for idx_second_clf, second_clf in enumerate(self.clfs):
                    t_test, p_value = stats.ttest_rel(data[:, idx_first_clf], data[:, idx_second_clf])
                    t_student_matrix[idx_first_clf, idx_second_clf] = t_test
                    p_matrix[idx_first_clf, idx_second_clf] = p_value
                    tf_matrix[idx_first_clf, idx_second_clf] = np.mean(data[:, idx_first_clf]) > np.mean(data[:, idx_second_clf])
                    alpha_matrix[idx_first_clf, idx_second_clf] = p_value < alpha
                    advantage_matrix[idx_first_clf, idx_second_clf] = np.logical_and(tf_matrix[idx_first_clf, idx_second_clf], alpha_matrix[idx_first_clf, idx_second_clf])

            save_file_path = f'tests/test_results/{filename}_test_result.txt'
            f = open(save_file_path, 'w')
            f.write(f'Test statystyczny dla zbioru: {filename}.csv\n'
                    f'Macierz wartości t_studenta:\n {t_student_matrix}\n\n\n'
                    f'Macierz wartości p:\n {p_matrix}\n\n\n'
                    f'Macierz true-false:\n {tf_matrix}\n\n\n'
                    f'Macierz wartości alpha:\n {alpha_matrix}\n\n\n'
                    f'Macierz przewagi:\n {advantage_matrix}\n\n\n')


