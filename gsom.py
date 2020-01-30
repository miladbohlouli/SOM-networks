import numpy as np
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

class gsom_node:
    def __init__(self, parent, id, connected_nodes, level=None, wins=None):
        self.parent = parent
        self.id = id
        self.wins = wins
        self.AE = 0
        self.is_boundry = True
        self.connected_nodes = connected_nodes
        self.weights = None
        self.level = level
        # self.pos = pos

    def weight_initializer(self, neighbors=None, X=None):
        if self.parent.parent == None:
            self.weights = X[np.random.choice(np.arange(X.shape[0]))]
        else:
            nei_weights = []
            for nei in neighbors:
                nei_weights.append(nei.weights)
            nei_weights = np.array(nei_weights)
            # print(nei_weights)
            self.weights = nei_weights[np.argmax(np.linalg.norm(nei_weights))]\
                    + np.abs(np.random.random(nei_weights[0].shape))


class gsom:
    def __init__(self, X, spread_factor, learning_rate, gamma):
        self.nodes = []
        self.node_counter = 0
        self.spread_factor = spread_factor
        self.GT = -X.shape[1] * np.log(self.spread_factor)
        # print(self.GT)
        # input()
        self.learning_rate = learning_rate
        self.neighbor_strength = 0.1
        self.gamma = gamma
        self.linkage_matrix = []

        root_node = gsom_node(parent=None, id='root_node', connected_nodes=4, level=0)
        # init_pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for i in range(4):
            new_node = gsom_node(parent=root_node, id=self.node_counter, connected_nodes=2)
            new_node.weight_initializer(X=X)
            new_node.level = new_node.parent.level + 1
            self.nodes.append(new_node)
            self.node_counter += 1


    def get_boundries(self):
        boundary_weights = []
        boundary_nodes = []
        for node in self.nodes:
            if node.connected_nodes != 4:
                boundary_nodes.append(node)
                boundary_weights.append(node.weights)
        boundary_weights = np.array(boundary_weights)
        boundary_nodes = np.array(boundary_nodes)
        return boundary_nodes, boundary_weights

    # def find_winners(self, X):
    #     boundary_nodes, boundary_weights = self.get_boundries()
    #     dists = cdist(boundary_weights, X)
    #     AEs = np.sum(dists, axis=1)
    #     winners = np.argmin(dists, axis=0)
    #     win_counts = np.zeros(boundary_nodes.shape[0])
    #
    #     for i in range(boundary_nodes.shape[0]):
    #         boundary_nodes[i].AE = AEs[i]
    #         boundary_nodes[i].wins = np.sum(winners == i)
    #         win_counts[i] = boundary_nodes[i].wins
    #
    #     return boundary_nodes[np.argmax(win_counts)]

    def find_winners(self, X):
        weights = []
        for node in self.nodes:
            node.AE = 0
            weights.append(node.weights)

        dists = cdist(weights, X)
        winners = np.argmin(dists, axis=0)
        for i in range(dists.shape[0]):
            self.nodes[i].AE = np.sum(dists[0, winners==i])

        max_ae = self.nodes[0].AE
        max_ae_node = self.nodes[0]
        for node in self.nodes:
            if node.AE > max_ae:
                max_ae = node.AE
                max_ae_node = node

        while max_ae_node.connected_nodes == 4:
            max_ae_node.AE /= 2
            neighbors = self.get_childs(max_ae_node)
            for node in neighbors:
                node.AE *= (1 + self.gamma)

            max_ae = self.nodes[0].AE
            max_ae_node = self.nodes[0]
            for node in self.nodes:
                if node.AE > max_ae:
                    max_ae = node.AE
                    max_ae_node = node
        # print(max_ae_node.AE)
        return max_ae_node

    def get_neighbors(self, node):
        neighbors = []
        for n in self.nodes:
            if n.parent == node.parent:
                neighbors.append(n)
        return np.array(neighbors)

    def get_childs(self, node):
        childs = []
        for n in self.nodes:
            if n.parent == node:
                childs.append(n)
        return np.array(childs)

    # def expand_node(self, node):
    #     # print('node {} will expand, it has {} connected nodes, and will have {} neighbors'.format(node.id, node.connected_nodes, 4-node.connected_nodes))
    #     for i in range(4 - node.connected_nodes):
    #         node.connected_nodes += 1
    #         new_node = gsom_node(parent=node, id=self.node_counter, connected_nodes=1)
    #         neighbors = self.get_neighbors(node)
    #         new_node.weight_initializer(neighbors=neighbors)
    #         self.nodes.append(new_node)
    #         self.node_counter += 1
    #     pass

    def expand_node(self, node):
        # print('node {} will expand, it has {} connected nodes, and will have {} neighbors'.format(node.id, node.connected_nodes, 4-node.connected_nodes))
        for i in range(4 - node.connected_nodes):
            node.connected_nodes += 1
            new_node = gsom_node(parent=node, id=self.node_counter, connected_nodes=1)
            neighbors = self.get_neighbors(node)
            new_node.weight_initializer(neighbors=neighbors)
            new_node.level = new_node.parent.level + 1
            self.nodes.append(new_node)
            self.node_counter += 1
        pass

    def update_weights(self, X, phase):
        weights = []
        for node in self.nodes:
            weights.append(node.weights)
        weights = np.array(weights)

        dists = cdist(X, weights)
        winners = np.argmin(dists, axis=1)

        learning_rate = self.learning_rate
        for i in range(winners.shape[0]):
            self.nodes[winners[i]].weights += learning_rate * (X[i] - self.nodes[winners[i]].weights)
            if phase == 'growth':
                neighbors = self.get_neighbors(self.nodes[winners[i]])
                for nei in neighbors:
                    nei.weights += learning_rate * self.neighbor_strength * (X[i] - nei.weights)

    def plot_clusters(self, X, y_pred):
        plt.clear()
        plt.pause(0.001)
        for c_id in np.unique(y_pred):
            this_cluster = X[y_pred == c_id]
            plt.scatter(this_cluster[:, 0], this_cluster[:, 1], s=4)
        plt.draw()
        plt.pause(0.001)


    def train(self, X, epochs):
        fig, ax = plt.subplots(1, 1)
        for epoch in range(epochs):
            self.update_weights(X, 'growth')
            winner = self.find_winners(X)
            self.expand_node(winner)
            # bound, b_w = self.get_boundries()
            self.update_weights(X, 'smoothing')
            # y_pred = self.assign_clusters(X)
            #
            # ax.clear()
            # for c_id in np.unique(y_pred):
            #     this_cluster = X[y_pred == c_id]
            #     # print('cluster {} has {} samsples'.format(self.nodes[c_id].id, this_cluster.shape[0]))
            #     plt.scatter(this_cluster[:, 0], this_cluster[:, 1], s=4)
            # plt.draw()
            # plt.pause(0.5)
            # input()

            # print(len(self.nodes))
        # counter = 0
        # while self.get_boundries()[0].shape[0] != X.shape[0]:
        #     self.update_weights(X, 'growth')
        #     winner = self.find_winners(X)
        #     self.expand_node(winner)
        #     self.update_weights(X, 'smoothing')
        #     if counter % 10 == 0:
        #         print(counter)
        #     counter += 1

    def assign_clusters(self, X):
        boundary_nodes, boundary_weights = self.get_boundries()
        # print('assign_clusters')
        # print(boundary_weights.shape)
        dists = cdist(boundary_weights, X)
        winners = np.argmin(dists, axis=0)

        labels = np.zeros(winners.shape[0], dtype=np.int)
        unique = np.unique(winners)
        for c_id in unique:
            labels[winners == c_id] = boundary_nodes[c_id].id

        return labels

    def get_level_nodes(self, level):
        level_nodes = np.array([])
        for node in self.nodes:
            if node.level == level:
                level_nodes = np.append(level_nodes, node)
        return level_nodes


    def get_level_cluster(self, this_level_nodes):
        parents = np.array([])
        for node in this_level_nodes:
            parents = np.append(parents, node.parent.id)

        clusters = []
        for parent_id in np.unique(parents):
            this_cluster = []
            for node in this_level_nodes:
                if node.parent.id == parent_id:
                    this_cluster.append(node)
            clusters.append(this_cluster)

        return clusters

    def get_node_by_id(self, node_id):
        for node in self.nodes:
            if node.id == node_id:
                return node

    def get_parents(self, node_id):
        tmp = self.get_node_by_id(node_id)
        parents = []
        while tmp.id != 'root_node':
            parents.append(tmp.id)
            tmp = tmp.parent
        return np.array(parents)

    def get_cluster_samples(self, X, cluster_id):
        y_pred = self.assign_clusters(X)
        this_cluster_samples = []

        for i in range(X.shape[0]):
            if cluster_id in self.get_parents(y_pred[i]):
                this_cluster_samples.append(X[i])

        return np.array(this_cluster_samples)


    def calc_cluster_distance(self, X, cluster_id_1, cluster_id_2, mode='single_linkage'):
        cluster_1 = self.get_cluster_samples(X, cluster_id_1)
        cluster_2 = self.get_cluster_samples(X, cluster_id_2)

        if cluster_1.shape[0] == 0 or cluster_2.shape[0] == 0:
            return 10

        if mode == 'single_linkage':
            return np.min(cdist(cluster_1, cluster_2))

    def linkage_matrix_4(self, X):
        # self.print_net()
        boundary_nodes, boundary_weights = self.get_boundries()
        levels = np.array([])
        cluster_dict = dict()
        cluster_ids = dict()
        cluster_counter = boundary_nodes.shape[0]
        linkage = []
        dist = 1
        b_node_id = dict()

        for i in range(boundary_nodes.shape[0]):
            b_node_id[boundary_nodes[i].id] = i

        for node in self.nodes:
            cluster_dict[node.id] = 0
        cluster_dict['root_node'] = 0

        for node in boundary_nodes:
            levels = np.append(levels, node.level)
            cluster_dict[node.id] = 1
        max_level = np.max(levels)
        curr_level = max_level

        while curr_level >= 0:
            this_level_nodes = self.get_level_nodes(curr_level)
            this_level_clusters = self.get_level_cluster(this_level_nodes)

            for cluster in this_level_clusters:
                if len(cluster) == 1:
                    pass

                if len(cluster) == 2:
                    clust_id = []
                    for node in cluster:
                        if node.id in cluster_ids:
                            clust_id.append(cluster_ids[node.id])
                        else:
                            clust_id.append(b_node_id[node.id])
                    # dist = self.calc_cluster_distance(X, cluster[0].id, cluster[1].id)
                    linkage.append([clust_id[0], clust_id[1], dist, cluster_dict[cluster[0].id] + cluster_dict[cluster[1].id]])
                    cluster_dict[cluster[0].parent.id] += cluster_dict[cluster[0].id] + cluster_dict[cluster[1].id]
                    cluster_ids[cluster[0].parent.id] = cluster_counter
                    cluster_counter += 1
                    dist += 1

                if len(cluster) == 3:
                    clust_id = []
                    for node in cluster:
                        if node.id in cluster_ids:
                            clust_id.append(cluster_ids[node.id])
                        else:
                            clust_id.append(b_node_id[node.id])

                    # dists = np.array([])
                    # dists = np.append(dists, self.calc_cluster_distance(X, cluster[0].id, cluster[1].id))
                    # dists = np.append(dists, self.calc_cluster_distance(X, cluster[1].id, cluster[2].id))
                    # dists = np.append(dists, self.calc_cluster_distance(X, cluster[0].id, cluster[2].id))
                    # dist = np.min(dists)


                    linkage.append([clust_id[0], clust_id[1], dist, cluster_dict[cluster[0].id] + cluster_dict[cluster[1].id]])
                    linkage.append([cluster_counter, clust_id[2], dist, cluster_dict[cluster[0].id] + cluster_dict[cluster[1].id]])

                    cluster_dict[cluster[0].parent.id] += cluster_dict[cluster[0].id] + cluster_dict[cluster[1].id]
                    cluster_ids[cluster[0].parent.id] = cluster_counter + 1

                    cluster_counter += 2
                    dist += 1


                if len(cluster) == 4:
                    clust_id = []
                    for node in cluster:
                        if node.id in cluster_ids:
                            clust_id.append(cluster_ids[node.id])
                        else:
                            clust_id.append(b_node_id[node.id])

                    linkage.append([clust_id[0], clust_id[1], dist, cluster_dict[cluster[0].id] + cluster_dict[cluster[1].id]])
                    cluster_dict[cluster[0].parent.id] += cluster_dict[cluster[0].id] + cluster_dict[cluster[1].id]

                    linkage.append([clust_id[2], clust_id[3], dist, cluster_dict[cluster[2].id] + cluster_dict[cluster[3].id]])
                    cluster_dict[cluster[2].parent.id] += cluster_dict[cluster[2].id] + cluster_dict[cluster[3].id]

                    linkage.append([cluster_counter, cluster_counter+1, dist, cluster_dict[cluster[0].parent.id]])
                    cluster_counter += 2
                    dist += 1


            curr_level -= 1

        linkage = np.array(linkage, dtype=np.double)
        print(linkage)
        dendrogram(linkage)
        plt.show()

    def single_tone(self, a):
        for result in a[:, 3]:
            if result not in a[:, 0]:
                print('this cluster {} did not used in first'.format(result))
            if result not in a[:, 1]:
                print('this cluster {} did not used in second'.format(result))

    def print_net(self):
        boundary_nodes, boundary_weights = self.get_boundries()
        for node in boundary_nodes:
            # print(node.id, end='->')
            tmp = node
            while tmp.id != 'root_node':
                print(tmp.id, end='->')
                tmp = tmp.parent
            print()
