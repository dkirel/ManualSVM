import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np


class SVMClassifier:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {0: 'r', 1: 'b'}

        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):
        self.data = data

        # { ||w||: {w, b} }
        opt_dict = {}

        transforms = [[-1, -1], [-1, 1], [1, -1], [1, 1]]

        all_data = []
        for y in self.data:
            for featureset in self.data[y]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = max(all_data)
        all_data = None

        step_sizes = [self.max_feature_value*0.1,
                      self.max_feature_value*0.01,
                      self.max_feature_value*0.001]

        # more expensive
        b_range_multiple = 5

        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False

            while not optimized:
                for b in np.arange(-1*self.max_feature_value*b_range_multiple,
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transform in transforms:
                        w_t = w*tranform
                        valid_option = True

                        for y in self.data:
                            for x in self.data[y]:
                                if y*(np.dot(w_t, x)) + b < 1:
                                      valid_option = False
                                      break

                            if not valid_option:
                                break

                        if found_option:
                           opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0:
                    optimized = True
                    print('Optimized a Step.')
                else:
                    w -= step

                norms = sorted([n for n in opt_dict])
                opt_choice = opt_dict[norms[0]]

                self.w = opt_choice[0]
                self.b = opt_choice[1]


                latest_optimum = opt_choice[0][0]
                    
        
    def predict(self, features):
        prediction = np.sigh(np.dot(np.array(features), self.w) + self.b)

        if prediction != 0 and self.visualization:
            self.ax.scatter(featuers[0], features[1], s=200, marker=*, c.self.colors[precition])

        return prediction

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i] for x in data_dict[y] for y in data_dict]

        def hyperplane(x, w, b, v):
            return (-w[0]*x - b + v)/
          
style.use('ggplot')

data_dict = {
    -1: np.array([[1, 7], [2, 8], [3, 8]]),
    1: np.array([[5, 1], [6, -1], [7, 3]])
}


