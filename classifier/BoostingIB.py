from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from classifier.errors.EkError import EkError
from classifier.metrics.GeoMean import GeoMean
from sklearn.base import clone

# BoostingIB classifier by Patryk Nawrocki

class Boosting_IB(BaseEstimator, ClassifierMixin):

    def __init__(self, repeats=1):
        self.top_clf = None
        self.g = 0
        self.alpha = 0
        self.base_clf_list = [
            GaussianNB(),
            DecisionTreeClassifier(),
            KNeighborsClassifier(),
            RandomForestClassifier(),
            SVC()
        ]
        self.base_clf_list_with_repeats = []
        for repeat in range(repeats):
            for idx_clf in range(len(self.base_clf_list)):
                cloned_clf = clone(self.base_clf_list[idx_clf])
                self.base_clf_list_with_repeats.append(cloned_clf)

        self.clf_predicts = []
        self.weights = np.ones((len(self.base_clf_list_with_repeats)))
        self.clfs_after_fit = []
        self.errors = []

        # print(len(self.base_clf_list_with_repeats))

    def fit(self, x, y):
        no_of_random_train_samples = int(np.ceil(x.shape[0]/2))

        #print(no_of_random_train_samples)

        for idx_clf, classifier in enumerate(self.base_clf_list_with_repeats):
            random_train_samples = np.random.choice(x.shape[0], no_of_random_train_samples)

            #print(random_train_samples)
            #print(random_test_samples)
            x_train, y_train = x[random_train_samples], y[random_train_samples]

            classifier.fit(x_train, y_train)
            predicted_value = classifier.predict(x)
            clf_score = balanced_accuracy_score(y_pred=predicted_value, y_true=y)
            error = EkError().calc_ek_error(y=y, predicted_y=predicted_value)
            self.errors.append(error)
            if 0 < error < 0.5:
                self.alpha = np.log((1-error)/error)
                g_mean = GeoMean().geo_mean(pred=predicted_value, real=y)
                if g_mean > self.g:
                    self.g = g_mean
                    self.top_clf = classifier
                self.weights[idx_clf] = self.weights[idx_clf] * np.exp(self.alpha)
                self.weights[idx_clf] = self.weights[idx_clf] / np.sum(self.weights)
            else:
                self.alpha = 0
                self.weights[idx_clf] = 1
            self.clfs_after_fit.append(classifier)
        #print(self.weights)
        #print(len(self.weights))
        return self

    def predict(self, x):
        result_array = np.zeros((x.shape[0]))

        for index_weight, weight in enumerate(self.weights):
            predicted_value = self.clfs_after_fit[index_weight].predict(x)
            #print(predicted_value)
            for idx, p in enumerate(predicted_value):
                if p == 0:
                    p = -1
                result_array[idx] = result_array[idx] + (p*weight)

        for idx, p in enumerate(result_array):
            if p < 0:
                result_array[idx] = 0

       #print(result_array)
        #print(np.sign(result_array))

        return np.sign(result_array)


    def get_len_of_base_clfs(self):
        return len(self.base_clf_list)
