import numpy as np


class EimbError:
    def __init__(self):
        pass

    def calc_eimb_error(self, y, predicted_y, weights):

        #print(f'y:{y}')
        #print(f'predicted:{predicted_y}')
        result = np.zeros(y.shape[0], float)
        for index, element in enumerate(y):
            if y[index] == predicted_y[index]:
                result[index] = 1
            else:
                result[index] = 0

        score = (np.sum(result) / len(result))
        #print(result)
        #print(score)
        # result = np.sum(weights * (predicted_y != y)) * 1/np.sum(predicted_y) + np.sum(weights * (predicted_y != y)) * 1/np.sum(y)
        return score
