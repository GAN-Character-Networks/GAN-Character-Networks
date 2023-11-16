import networkx as nx


class GraphManager:
    r"""
    Gestionnaire de graphe.

    Permet de créer un graphe pondéré ou non pondéré basé sur des cooccurrences.

    Exemple d'utilisation:
    graph_manager = GraphManager(weighted=True)
    cooccurrences = [("Sheldon", "Arthur"), ("Arthur", "Bilal"), ("Sheldon", "Arthur")]
    graph_manager.add_cooccurrences(cooccurrences)
    graph_manager.save_graph_to_graphml("cooccurrence_graph.graphml")
    """

    def __init__(self, weighted=False):
        """
        Initialise un nouveau gestionnaire de graphe.

        :param weighted: Booléen indiquant si le graphe doit être pondéré ou non.
        """
        self.graph = nx.Graph()
        self.weighted = weighted

    def add_cooccurrences(self, cooccurrences):
        """
        Ajoute ou met à jour les cooccurrences dans le graphe.

        Si weighted est True, les arêtes sont pondérées par le nombre de cooccurrences.
        Sinon, toutes les arêtes ont le même poids (non pondéré).

        :param cooccurrences: Liste de tuples représentant les cooccurrences.
        """
        for cooccurrence in cooccurrences:
            if self.weighted:
                if self.graph.has_edge(*cooccurrence):
                    # Augmenter le poids si l'arête existe déjà
                    self.graph[cooccurrence[0]][cooccurrence[1]]["weight"] += 1
                else:
                    # Ajouter une nouvelle arête avec un poids initial
                    self.graph.add_edge(
                        cooccurrence[0], cooccurrence[1], weight=1
                    )
            else:
                # Ajouter une arête sans poids
                self.graph.add_edge(*cooccurrence)

    def save_graph_to_graphml(self, filename):
        """
        Sauvegarde le graphe actuel au format GraphML.

        :param filename: Nom du fichier pour sauvegarder le graphe.
        """
        nx.write_graphml(self.graph, filename)
