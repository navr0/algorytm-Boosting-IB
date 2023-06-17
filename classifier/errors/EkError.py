class EkError:
    def __init__(self):
        self.errors = []

    def calc_ek_error(self, y, predicted_y):
        total_sum = 0

        for index in range(len(y)):
            if predicted_y[index] == y[index]:
                total_sum = total_sum + 1
            else:
                predicted_y[index] = predicted_y[index]
        error_value = 1 - total_sum / len(y)
        #print('Error: %.3f' % error_value)
        return error_value
