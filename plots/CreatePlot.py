import numpy as np
from matplotlib import pyplot as plt


class CreatePlot:
    def __init__(self, setname, num_of_repeats, mean_plots, std_plots=None):
        self.num_of_repeats = num_of_repeats
        self.setname = setname
        self.mean_plots = mean_plots
        self.std_plots = std_plots
        self.title = f'{self.setname} - zależność accuracy od\nliczby klasyfikatorów bazowych - BoostingIB'
        self.x_label = f'Liczba klasyfikatorow bazowych'
        self.y_label = f'Wartosci srednie'
        self.path_name = f'plots/plots_images/plot_n_{self.setname}.jpg'
        self.no_clfs = np.zeros((len(self.num_of_repeats)))
        for i in range(len(self.num_of_repeats)):
            self.no_clfs[i] = self.num_of_repeats[i] * 5

    def generate_plot(self):
        plt.scatter(self.no_clfs, self.mean_plots)
        plt.grid()
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        print(self.num_of_repeats)
        print(self.mean_plots)
        plt.savefig(self.path_name)

        plt.show()





