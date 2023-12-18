import networkx as nx


class GraphManager:
    r"""
    Gestionnaire de graphe.

    Permet de créer un graphe pondéré ou non pondéré basé sur des cooccurrences.

    Exemple d'utilisation:
    graph_manager = GraphManager(weighted=True)
    cooccurrences = [(["Sheldon", "Harry"], ["Arthur", "Amalvy"]), (["Arthur", "Amalvy", ["Bilal", "Marzougue"]), (["Sheldon", "Harry"], ["Arthur", "Amalvy"])]
    graph_manager.add_cooccurrences(cooccurrences)
    graphml = graph_manager.generate_graph()
    graph_manager.save_graph_to_graphml("cooccurrence_graph.graphml")
    """

    def __init__(self, weighted: bool = False):
        """
        Initialise un nouveau gestionnaire de graphe.

        :param weighted: Booléen indiquant si le graphe doit être pondéré ou non.
        """
        self.graph = nx.Graph()
        self.weighted = weighted

    def add_cooccurrences(
        self, cooccurrences: list[tuple[list[str], list[str]]]
    ) -> None:
        """
        Ajoute ou met à jour les cooccurrences dans le graphe.

        Si weighted est True, les arêtes sont pondérées par le nombre de cooccurrences.
        Sinon, toutes les arêtes ont le même poids (non pondéré).

        :param cooccurrences: Liste de tuples de listes contenant les alias des personnages impliqués dans une cooccurrence.
        """
        aliases = {}
        for cooccurrence in cooccurrences:
            group_1, group_2 = cooccurrence
            aliases[group_1[0]] = group_1
            aliases[group_2[0]] = group_2

            cooccurrence = tuple([group_1[0], group_2[0]])
            if self.weighted:
                if self.graph.has_edge(*cooccurrence):
                    # Augmenter le poids si l'arête existe déjà
                    self.graph[cooccurrence[0]][cooccurrence[1]]["weight"] += 1
                else:
                    # Ajouter une nouvelle arête avec un poids initial
                    self.graph.add_edge(*cooccurrence, weight=1)
            else:
                # Ajouter une arête sans poids
                self.graph.add_edge(*cooccurrence)

            # Ajout des alias des personnages impliqués dans la cooccurrence
            for alias in cooccurrence:
                self.graph.nodes[alias]["names"] = ";".join(aliases[alias])

        print(self.graph)

    def generate_graph(self) -> nx.graphml:
        """
        Génère le graphe à partir des cooccurrences ajoutées.

        :return: Le graphe généré.
        """
        return nx.generate_graphml(self.graph)

    def save_graph_to_graphml(self, filename):
        """
        Sauvegarde le graphe actuel au format GraphML.

        :param filename: Nom du fichier pour sauvegarder le graphe.
        """
        nx.write_graphml(self.graph, filename)
