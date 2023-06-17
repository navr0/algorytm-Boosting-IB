import os
from datetime import datetime

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

from classifier.BoostingIB import Boosting_IB
from plots.CreatePlot import CreatePlot


class RunBoostingIB:
    def __init__(self, repeats):
        self.repeats = repeats

    def run(self):
        dir_path = "datasets"
        csv_files = os.listdir(dir_path)
        big_matrix = np.zeros((len(csv_files)))
        b_ib_scores = np.zeros((len(csv_files), 10))
        time_start = datetime.now()
        for csv_idx, filename in enumerate(csv_files):
            file_path = f'{dir_path}/{filename}'
            current_data = np.loadtxt(file_path, delimiter=',')

            x = current_data[:, :-1]
            y = current_data[:, -1]

            # print(x.shape)
            # print(y.shape)

            rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2)

            repeats_mean_array = []
            for repeat in self.repeats:
                filename = filename.split(".csv")[0]
                save_file_path = f'result_directory/csv_files/{filename}_repeats{repeat}.csv'
                f = open(save_file_path, 'w')
                f.write(f'rskf_idx;score\n')
                rskf_scores = []
                for rskf_index, (train_index, test_index) in enumerate(rskf.split(x, y)):
                    x_train, y_train = x[train_index], y[train_index]
                    x_test, y_test = x[test_index], y[test_index]
                clf = Boosting_IB(repeats=repeat)
                clf.fit(x_train, y_train)
                predicted_value = clf.predict(x_test)
                # print(f'Predicted value: {predicted_value}')
                score = balanced_accuracy_score(y_true=y_test, y_pred=predicted_value)
                print(f'score: {score}')
                f.write(f'{rskf_index};{score}\n')
                b_ib_scores[csv_idx, rskf_index] = score
                rskf_scores.append(score)
                repeats_mean_array.append(rskf_scores)
            CreatePlot(setname=filename, num_of_repeats=self.repeats, mean_plots=repeats_mean_array).generate_plot()
            time_stop = datetime.now()
            print(f'Czas startu algorytmu: {time_start}\nCzas zakończenia algorytmu: {time_stop}')

        print(b_ib_scores)
        time_end = datetime.now()
        print(f'Czas startu algorytmu: {time_start}\nCzas zakończenia algorytmu: {time_end}')

        pass