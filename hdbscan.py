import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def get_data():
        moons, _ = data.make_moons(n_samples=50, noise=0.05)
        blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25)
        test_data = np.vstack([moons, blobs])

        return test_data

    @staticmethod
    def plot_data(test_data, color='b'):
        sns.set_context('poster')
        sns.set_style('white')
        sns.set_color_codes()
        plot_kwds = {'alpha': 0.5, 's': 80, 'linewidths': 0}

        plt.scatter(test_data.T[0], test_data.T[1], color=color, **plot_kwds)
        plt.show()

    @staticmethod
    def get_dist(test_data):
        dist_mat = np.zeros(shape=(test_data.shape[0], test_data.shape[0]))
        for i in range(test_data.shape[0]):
            for j in range(test_data.shape[0]):
                dist_mat[i, j] = np.sqrt(np.sum([np.square(x-y) for x, y in zip(test_data[i, :], test_data[j, :])]))

        return dist_mat

    @staticmethod
    def core_dist(dist_mat, k=5):
        core_mat = np.zeros(shape=dist_mat.shape[0])
        for i in range(dist_mat.shape[0]):
            dist = np.sort(dist_mat[i, :])[::-1]
            core_mat[i] = dist[k-1]

        return core_mat

    @staticmethod
    def transform_space(dist_mat, core_mat):
        for i in range(dist_mat.shape[0]):
            for j in range(dist_mat.shape[0]):
                dist_mat[i, j] = np.amax([core_mat[i], core_mat[j], dist_mat[i, j]])

        return dist_mat


if __name__ == '__main__':
    data = Utils.get_data()
    Utils.plot_data(data)
    dist = Utils.get_dist(data)
    core_dist = Utils.core_dist(dist)
    dist = Utils.transform_space(dist, core_dist)