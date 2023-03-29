from nlns.problem.vrp import VRPInstance
import numpy as np

TEST_INSTANCE = {
    "depot": (0, 0),
    "nodes": [
        (-1, 1),
        (-1, 0.9),
        (1, 1),
        (1, -1),
    ],
    "demands": [
        3, 3, 3, 3
    ]
}


class TestVRPInstance:
    def test_init(self):
        VRPInstance(TEST_INSTANCE["depot"],
                    TEST_INSTANCE["nodes"],
                    TEST_INSTANCE["demands"])

    def test_adjacency_matrix(self):
        edge_encoding = {
            "node-node": 1,
            "node-node_knn": 2,
            "node-self": 3,
            "node-depot": 4,
            "node-depot_knn": 5,
            "depot-self": 6, }

        instance = VRPInstance(TEST_INSTANCE["depot"],
                               TEST_INSTANCE["nodes"],
                               TEST_INSTANCE["demands"])

        # test without neighbors
        adj = instance.adjacency_matrix(num_neighbors=0,
                                        edge_encoding=edge_encoding)

        assert adj[0, 0] == edge_encoding["depot-self"]
        assert (adj[1:, 0] == edge_encoding["node-depot"]).all()
        assert (np.diag(adj)[1:] == edge_encoding["node-self"]).all()

        # test fully connected
        adj = instance.adjacency_matrix(num_neighbors=-1,
                                        edge_encoding=edge_encoding)

        assert adj[0, 0] == edge_encoding["depot-self"]
        assert (adj[1:, 0] == edge_encoding["node-depot_knn"]).all()
        assert all((
            np.delete(adj[i], [0, i]) == edge_encoding["node-node_knn"]).all()
            for i in range(1, adj.shape[1]))
        assert (np.diag(adj)[1:] == edge_encoding["node-self"]).all()

        # test 1 neighbor
        adj = instance.adjacency_matrix(num_neighbors=1,
                                        edge_encoding=edge_encoding)

        assert adj[0, 0] == edge_encoding["depot-self"]
        assert adj[1, 2] == edge_encoding["node-node_knn"]
        assert adj[2, 1] == edge_encoding["node-node_knn"]
        assert (np.diag(adj)[1:] == edge_encoding["node-self"]).all()
