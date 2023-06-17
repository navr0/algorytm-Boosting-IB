import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier

# OLA clf by Tomasz Wąsik

class OLA(BaseEstimator, ClassifierMixin):

    def __init__(self, pool_classifiers):
        self.pool_classifiers = pool_classifiers
        self.predict_classifiers = []
        #self.predictions = []
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.predict_classifiers = []
        self.X_train = X_train
        self.y_train = y_train

        n_samples = int(np.ceil(X_train.shape[0] / len(self.pool_classifiers)))
        for classifier in self.pool_classifiers:
            random_samples = np.random.choice(X_train.shape[0], n_samples)
            X_train_bagging = X_train[random_samples]
            y_train_bagging = y_train[random_samples]
            classifier.fit(X_train_bagging, y_train_bagging)

        # for clf in self.pool_classifiers:
        #     clf.fit(X_train, y_train)

        return self

    def predict(self, X_test):
        X_train = self.X_train
        y_train = self.y_train

        for t in range(len(X_test)):
            predictions = []
            for i in self.pool_classifiers:
                prediction = i.predict([X_test[t]])[0]
                predictions.append(prediction)
                #print(predictions)  #0i1

            if all(prediction == predictions[0] for prediction in predictions):
                self.predict_classifiers.append(predictions[0])
            else:
                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(X_train, y_train)

                optimal = knn.kneighbors([X_test[t]], return_distance=False)[0]  #zwracane są indeksy najbliższych sąsiadów dla danej próbki X_test[t]
                #print(optimal) #[int int int]                               #bierzemy pierwszy zestaw elementów
                ola_scores = []
                for j in self.pool_classifiers:
                    correct_predictions = sum(j.predict(X_train[optimal]) == y_train[optimal]) #suma prawidłowych przewidywań dla próbek optymalnego sąsiedztwa
                    #print(correct_predictions) #0-3

                    ola_score = correct_predictions / len(y_train[optimal]) #ocena ola, czyli stosunek poprawnych predykcji do liczby próbek naszego optymalnego sąsiedztwa
                    ola_scores.append(ola_score)
                    #print(ola_scores) #0.33,0.66,1.0
                    #exit()

                predict_classifier_idx = np.argmax(ola_scores)
                self.predict_classifiers.append(self.pool_classifiers[predict_classifier_idx].predict([X_test[t]])[0])

        sum_array = np.zeros(len(X_test))
        for clf in self.pool_classifiers:
            predicted_array = clf.predict(X_test)
            sum_array += predicted_array

        sum_array /= len(self.pool_classifiers)
        final_predictions = np.zeros(len(X_test))
        final_predictions[sum_array > 0.5] = 1.0

        average_predictions = (final_predictions + self.predict_classifiers) / 2
        results = np.round(average_predictions)


        return results
        #return self.predict_classifiers