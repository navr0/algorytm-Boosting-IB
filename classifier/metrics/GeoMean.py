import numpy as np

#Sensitivity_And_Specifity


class GeoMean:
    def __init__(self):
        pass

    def geo_mean(self, pred, real):
        #wyliczenie:
        # true positive (predykcja = 1, wynik = 1),
        # true negative (predykcja = 0, wynik = 0),
        # false positive (predykcja = 1, wynik = 0),
        # false negative (predykcja = 0, wynik = 1)
        tp = np.sum(np.logical_and(pred == 1, real == 1))
        tn = np.sum(np.logical_and(pred == 0, real == 0))
        fp = np.sum(np.logical_and(pred == 1, real == 0))
        fn = np.sum(np.logical_and(pred == 0, real == 1))

        #czułość
        sensitivity = tp/(tp+fn)

        #swoistość
        specificity = tn/(fp+tn)

        #g_mean
        g_mean = np.sqrt(sensitivity*specificity)
        return g_mean