import os
import networkx as nx
from vroom.GraphManager import GraphManager


class TestGraphManager:
    """
    Tests unitaires pour la classe GraphManager.
    """

    def setup_method(self):
        """
        Initialise un nouveau gestionnaire de graphe et ajoute des cooccurrences.
        """
        # Gestionnaire pour graphe pondéré
        self.weighted_graph_manager = GraphManager(weighted=True)
        self.cooccurrences = [
            (
                ["Sheldon", "Arthur", "S.A", "José au moine"],
                ["Bilal Merzougue", "Bilal", "B.M", "Pointe à pitre"],
            ),
            (
                ["François Hollande", "François", "F.H", "François le grand"],
                ["Bilal Merzougue", "Bilal", "B.M", "Pointe à pitre"],
            ),
            (
                ["Macron", "Emmanuel Macron", "E.M", "Manu"],
                ["Bilal Merzougue", "Bilal", "B.M", "Pointe à pitre"],
            ),
            (
                ["Sheldon", "Arthur", "S.A", "José au moine"],
                ["François Hollande", "François", "F.H", "François le grand"],
            ),
        ]
        self.weighted_graph_manager.add_cooccurrences(self.cooccurrences)
        self.weighted_graph_manager.save_graph_to_graphml(
            "weighted_cooccurrence_graph.graphml"
        )

        # Gestionnaire pour graphe non pondéré
        self.unweighted_graph_manager = GraphManager(weighted=False)
        self.unweighted_graph_manager.add_cooccurrences(self.cooccurrences)
        self.unweighted_graph_manager.save_graph_to_graphml(
            "unweighted_cooccurrence_graph.graphml"
        )

    def test_weighted_graph_saved(self):
        """
        Vérifie que le graphe pondéré a bien été sauvegardé.
        """
        assert os.path.exists("weighted_cooccurrence_graph.graphml")

    def test_unweighted_graph_saved(self):
        """
        Vérifie que le graphe non pondéré a bien été sauvegardé.
        """
        assert os.path.exists("unweighted_cooccurrence_graph.graphml")

    def test_graphml_format(self):
        """
        Vérifie que les graphes ont bien été sauvegardés au format GraphML.
        """
        assert (
            nx.read_graphml("weighted_cooccurrence_graph.graphml") is not None
        )
        assert (
            nx.read_graphml("unweighted_cooccurrence_graph.graphml") is not None
        )

    def test_graphml_content(self):
        """
        Vérifie que les graphes ont bien été sauvegardés avec le contenu attendu.
        """
        weighted_graph = nx.read_graphml("weighted_cooccurrence_graph.graphml")
        unweighted_graph = nx.read_graphml(
            "unweighted_cooccurrence_graph.graphml"
        )
        assert weighted_graph.edges == self.weighted_graph_manager.graph.edges
        assert (
            unweighted_graph.edges == self.unweighted_graph_manager.graph.edges
        )

    def teardown_method(self):
        """
        Supprime les fichiers de graphe.
        """
        os.remove("weighted_cooccurrence_graph.graphml")
        os.remove("unweighted_cooccurrence_graph.graphml")
