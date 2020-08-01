import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib


class Perceptron:
    X = 0
    w = 0
    y = 0
    SquarederrorCount = [0]
    iterations = 0
    epochNumber = 0
    RMSErrorFunction = []

    nocache_metabolic_cost = [0, 0]

    transient_w = 0
    consolidation_iteration = [0]
    con_threshold = 0
    cache_metabolic_cost = [0, 0]
    maintenance_cost = 0
    consolidation_cost = 0
    maintenance_costs = []
    consolidation_costs = []

    def __init__(self, learning_rate, features, instances, con_thresholds, c):
        self.learning_rate = learning_rate
        self.features = features
        self.instances = instances
        self.con_thresholds = con_thresholds
        self.metabolic_df = pd.DataFrame()
        self.c = c

    def generate_data(self, p):
        X = np.random.binomial(1, p, (self.features, self.instances))
        for j in range(X.shape[0]):
            for i in range(X.shape[1]):
                if X[i, j] == 0:
                    X[i, j] = -1

        thresholds = np.ones((self.instances, 1))
        self.X = np.hstack((X, thresholds))

        self.y = np.random.binomial(1, 0.5, (self.instances, 1))

        original_weights = np.random.normal(0, 0.01, (self.X.shape[1], 1))
        joblib.dump(original_weights, 'original_weights.sav')
        self.w = original_weights

    @staticmethod
    def heaviside(a):
        if a > 0:
            f = 1
        else:
            f = 0
        return f

    def activation(self, i):
        return np.dot(self.X[i], self.w)

    def error(self, f, i):
        e = self.y[i] - f
        self.SquarederrorCount[self.epochNumber] += e**2
        return e

    def learning_rule(self, e, i):
        self.iterations += 1
        for n in range(len(self.w)):
            adjustment = self.learning_rate * e * self.X[i][n]
            self.w[n] += adjustment
            self.nocache_metabolic_cost[self.iterations] += float(np.sqrt(adjustment ** 2))
        self.nocache_metabolic_cost.append(self.nocache_metabolic_cost[self.iterations])
        #self.iterations or -1

    def learning_rule_with_cache(self, e, i):
        self.iterations += 1
        for n in range(len(self.w)):
            adjustment = self.learning_rate * e * self.X[i][n]
            self.transient_w[n] += adjustment
            self.w[n] += adjustment

            if np.sqrt((self.transient_w[n])**2) > self.con_threshold:
                # self.cache_metabolic_cost.append((self.cache_metabolic_cost[-1] + np.linalg.norm(self.transient_w, ord=1)))
                self.cache_metabolic_cost[self.iterations] += np.linalg.norm(self.transient_w, ord=1)
                self.consolidation_cost += np.linalg.norm(self.transient_w, ord=1)
                self.consolidation_iteration.append(self.iterations)
                self.transient_w *= 0

        self.cache_metabolic_cost[self.iterations] += self.c * np.linalg.norm(self.transient_w, ord=1)
        self.maintenance_cost += self.c * np.linalg.norm(self.transient_w, ord=1)
        self.cache_metabolic_cost.append(self.cache_metabolic_cost[self.iterations])
        # self.iterations or -1

    def epoch(self):
        for i in range(self.X.shape[0]):
            a = self.activation(i)
            f = self.heaviside(a)
            e = self.error(f, i)
            self.learning_rule(e, i)

    def epoch_with_cache(self):
        for i in range(self.X.shape[0]):
            a = self.activation(i)
            f = self.heaviside(a)
            e = self.error(f, i)
            self.learning_rule_with_cache(e, i)

    def fit(self):
        original_weights = joblib.load('original_weights.sav')
        self.w = original_weights
        self.epochNumber = 1
        self.SquarederrorCount.append(0)
        self.epoch()
        while self.SquarederrorCount[self.epochNumber] != 0:
            self.epochNumber += 1
            self.SquarederrorCount.append(0)
            self.epoch()
        print(self.iterations)
        metabolic_col = pd.Series(data=self.nocache_metabolic_cost, index=range(len(self.nocache_metabolic_cost))).to_frame()
        self.metabolic_df = self.metabolic_df.merge(metabolic_col, how='outer', left_index=True, right_index=True)
        for item in self.SquarederrorCount:
            self.RMSErrorFunction.append(np.sqrt(item/self.instances))

        weights = self.w
        joblib.dump(weights, 'model.sav')

    def fit_with_cache(self):
        if self.nocache_metabolic_cost == [0, 0]:
            self.fit()

        for th in range(len(self.con_thresholds)):
            original_weights = joblib.load('original_weights.sav')
            self.w = original_weights
            self.con_threshold = float(self.con_thresholds[th])
            self.transient_w = self.w * 0

            self.epochNumber = 1
            self.SquarederrorCount = [0]
            self.cache_metabolic_cost = [0, 0]
            self.consolidation_iteration = [0]
            self.iterations = 0
            self.SquarederrorCount.append(0)
            self.epoch_with_cache()
            self.maintenance_cost = 0
            self.consolidation_cost = 0
            while self.SquarederrorCount[self.epochNumber] != 0:
                self.epochNumber += 1
                self.SquarederrorCount.append(0)
                self.epoch_with_cache()
            print(self.iterations)
            self.cache_metabolic_cost.append((self.cache_metabolic_cost[-1] + np.linalg.norm(self.transient_w, ord=1)))
            self.consolidation_cost += np.linalg.norm(self.transient_w, ord=1)
            self.consolidation_iteration.append(self.iterations)
            metabolic_col = pd.Series(data=self.cache_metabolic_cost).to_frame()
            # if no maintenance cost use consolidation_iteration as index above, also edit out learn_with_cace
            self.metabolic_df = self.metabolic_df.merge(metabolic_col, how='outer', left_index=True, right_index=True)
            self.maintenance_costs.append(self.maintenance_cost)
            self.consolidation_costs.append(self.consolidation_cost)

        if len(self.nocache_metabolic_cost) == 1:
            col_names = []
        else:
            col_names = ['No Cache']
        for col in self.con_thresholds:
            col_names.append('Threshold = ' + str(col))
        self.metabolic_df.columns = col_names
        self.metabolic_df = self.metabolic_df.fillna(method='ffill', axis=0)
        print(self.metabolic_df.head(15))
        print(self.consolidation_iteration)

        weights = self.w
        joblib.dump(weights, 'model.sav')

    def present_metabolic_data(self):
        print('\nRoot Mean Squared Error per epoch: ')
        for m in range(len(self.RMSErrorFunction)):
            print('Epoch {}: {}'.format(m, float(self.RMSErrorFunction[m])))

        original_weights = joblib.load('original_weights.sav')
        minimum = np.linalg.norm(self.w - original_weights, ord=1)
        efficiency_df = self.metabolic_df.div(minimum)

        fig, ax = plt.subplots(1, 1)
        ax.plot(range(len(self.RMSErrorFunction)), self.RMSErrorFunction)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('RMS Error')
        ax.set_title('Training curve')

        ax2 = self.metabolic_df.plot(title='Metabolic Cost with Consolidation Threshold')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Cumulative weight change')
        kwargs = {'linestyle': '--'}
        ax2.axhline(minimum, **kwargs)

        ax3 = efficiency_df.plot(title='Metabolic Efficiency with Consolidation Threshold')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Metabolic Cost /\nMinimum Cost')

        plt.show()

    def plot_metabolic_against_con_thresh(self):
        f = self.metabolic_df.iloc[-1]
        f = list(f.values)
        f.pop(0)
        x = list(self.con_thresholds)

        fig, ax = plt.subplots(1, 1)
        ax.plot(x, f, label='Total Cost')
        ax.set_xlabel('Consolidation Threshold')
        ax.set_ylabel('Metabolic Cost')
        ax.plot(x, self.maintenance_costs, label='Maintenance Cost')
        ax.plot(x, self.consolidation_costs, label='Consolidation Cost')
        plt.legend()

        plt.show()

    def test(self):
        model = joblib.load('model.sav')
        self.w = model
        self.iterations = 0
        self.errorCount = [0]
        predictions = []
        for i in range(self.X.shape[0]):
            self.iterations += 1
            a = self.activation(i)
            f = self.heaviside(a)
            predictions.append(f)
            self.error(f, i)
        print('\n\nTEST DATA')
        print('Root Mean Squared Error: ', np.sqrt(self.errorCount[0]/self.X.shape[0]))
        accuracy = accuracy_score(self.y, predictions)
        print('Accuracy: ', accuracy)
        print('Confusion Matrix: ')
        print(confusion_matrix(self.y, predictions))


def train_perceptron():
    np.random.seed(0)

    run.generate_data(0.5)

    run.fit()

    run.fit_with_cache()

    run.present_metabolic_data()

    run.plot_metabolic_against_con_thresh()

def test_perceptron():
    run.generate_data(0.5)

    run.test()


consolidation_thresholds = range(1, 21)
learning_rate = 1
features = 100
instances = 100
transient_maintenance_coeff = 0.05

run = Perceptron(learning_rate, features, instances, consolidation_thresholds, transient_maintenance_coeff)

train_perceptron()


#test_perceptron()













        











