import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
from scipy.cluster import hierarchy
from collections import deque


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def get_data():
        moons, _ = data.make_moons(n_samples=50, noise=0.1)
        blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25)
        test_data = np.vstack([moons, blobs])

        return test_data

    @staticmethod
    def plot_data(test_data, edges, transformed_distance, color='b'):
        sns.set_context('poster')
        sns.set_style('white')
        sns.set_color_codes()
        plot_kwds = {'alpha': 0.5, 's': 80, 'linewidths': 0}

        plt.scatter(test_data.T[0], test_data.T[1], color=color, **plot_kwds)

        for i in range(test_data.shape[0]):
            for j in range(test_data.shape[0]):
                if edges[i, j]:
                    plt.plot([test_data[i, 0], test_data[j, 0]], [test_data[i, 1], test_data[j, 1]], 'k-',
                             linewidth=transformed_distance[i, j])
        plt.show()

        return True

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
            distance = np.sort(dist_mat[i, :])
            core_mat[i] = distance[k-1]

        return core_mat

    @staticmethod
    def transform_space(dist_mat, core_mat):
        transformed_dist_mat = np.zeros(shape=(dist_mat.shape[0], dist_mat.shape[0]))
        for i in range(dist_mat.shape[0]):
            for j in range(dist_mat.shape[0]):
                transformed_dist_mat[i, j] = np.amax([core_mat[i], core_mat[j], dist_mat[i, j]])

        return transformed_dist_mat

    @staticmethod
    def prims_algorithm(dist_mat):
        not_visited = list(range(1, dist_mat.shape[0]))
        edges = np.zeros(shape=(dist_mat.shape[0], dist_mat.shape[0]))

        visited = [0]

        while len(not_visited):
            vertex = np.argmin(dist_mat[visited, :][:, not_visited])
            q, r = divmod(int(vertex), len(not_visited))
            q, r = visited[q], not_visited[r]
            edges[q, r] = dist_mat[q, r]
            visited.append(r)
            not_visited.remove(r)

        return edges

    @staticmethod
    def cluster_hierarchy(edges, num_points):
        out = np.zeros(shape=(num_points-1, 4))
        e_sorted = np.argsort(edges.flatten())[-(num_points - 1):]
        hierarchy_nodes = {i: [i, 1] for i in range(num_points)}
        hierarchy_clusters = {i: [i] for i in range(num_points)}
        hierarchy_tree = dict()
        curr_hierarchy = num_points - 1
        for i in range(e_sorted.shape[0]):
            q, r = divmod(e_sorted[i], num_points)
            s, t = hierarchy_nodes[q][1], hierarchy_nodes[r][1]
            out[i, 0], out[i, 1] = hierarchy_nodes[q][0], hierarchy_nodes[r][0]
            out[i, 2] = np.amax(edges[hierarchy_clusters[out[i, 0]], :][:, hierarchy_clusters[out[i, 1]]])
            out[i, 3] = s + t
            curr_hierarchy += 1
            for j in hierarchy_clusters[out[i, 0]]:
                hierarchy_nodes[j] = [curr_hierarchy, out[i, 3]]
            for k in hierarchy_clusters[out[i, 1]]:
                hierarchy_nodes[k] = [curr_hierarchy, out[i, 3]]
            hierarchy_clusters.update({curr_hierarchy: hierarchy_clusters[out[i, 0]] + hierarchy_clusters[out[i, 1]]})
            hierarchy_tree.update({curr_hierarchy: [[out[i, 0], s], [out[i, 1], t], out[i, 2]]})
        return out, hierarchy_tree, hierarchy_clusters

    @staticmethod
    def plot_dendrogram(out):
        dendrogram = hierarchy.dendrogram(out)
        plt.show()

        return dendrogram

    @staticmethod
    def condense_cluster_tree(hierarchy_tree, hierarchy_clusters, min_cluster_size, num_points):
        clusters_stabilities = dict()

        start = (num_points*2) - 2

        clusters_stack = deque()
        clusters_stack.append(start)

        while len(clusters_stack):
            i = clusters_stack.pop()
            v = hierarchy_tree[i]
            cluster_birth = 1/v[2]
            points_lambda = []

            while v[0][1] < min_cluster_size or v[1][1] < min_cluster_size:
                next_v = None
                if v[0][1] < min_cluster_size:
                    for _ in hierarchy_clusters[v[0][0]]:
                        points_lambda.append(1/v[2])
                else:
                    next_v = v[0][0]
                if v[1][1] < min_cluster_size:
                    for _ in hierarchy_clusters[v[1][0]]:
                        points_lambda.append(1/v[2])
                else:
                    next_v = v[1][0]
                v = hierarchy_tree[next_v]

            cluster_death_points_fall = len(hierarchy_clusters[i]) - len(points_lambda)
            cluster_death = 1/v[2]
            sum_stabilities = (cluster_birth - cluster_death) * cluster_death_points_fall
            sum_stabilities += -np.sum(np.array(points_lambda) - cluster_birth)
            clusters_stabilities.update({i: sum_stabilities})

            clusters_stack.append(v[0][0])
            clusters_stack.append(v[1][0])

        return clusters_stabilities

    @staticmethod
    def extract_clusters(hierarchy_clusters, clusters_stabilities):

        for k, v in hierarchy_clusters.items()[:-1]:
            pass

        return True


if __name__ == '__main__':
    data = Utils.get_data()
    dist = Utils.get_dist(data)
    core_dist = Utils.core_dist(dist)
    transformed_dist = Utils.transform_space(dist, core_dist)
    e = Utils.prims_algorithm(transformed_dist)
    plot = Utils.plot_data(data, e, transformed_dist)
    Z, tree, clusters = Utils.cluster_hierarchy(e, num_points=transformed_dist.shape[0])
    # dn = Utils.plot_dendrogram(Z)
    c_stabilities = Utils.condense_cluster_tree(tree, clusters, min_cluster_size=5, num_points=transformed_dist.shape[0])
    clusters = Utils.extract_clusters(clusters, c_stabilities)
