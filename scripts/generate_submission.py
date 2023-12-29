from vroom.baseline import get_cooccurences_with_aliases_and_gpt
from vroom.GraphManager import GraphManager
import pandas as pd
from tqdm import tqdm
from vroom.loggers import JSONLogger
import html
import os
"""
ID : un ID unique identifiant le chapitre correspondant au graphe fourni. L'id est de la forme {code du livre}{numéro du chapitre}. Les numéros des chapitres démarrent à 0. Les codes de livres sont paf et lca (pour Prélude à Fondation et Les Cavernes d'Acier, respectivement). Ainsi, le graphe du premier chapitre de Prélude à Fondation a pour ID paf0, et celui du dernier chapitre des Cavernes d'Acier a pour ID lca17.
graphml :: le graphe du chapitre identifié par la colonne ID, au format graphml. Chaque noeud peut être n'importe quelle chaîne de caractères, mais un attribut names doit être présent contenant les noms du personnages apparaissant durant le chapitre. Ces noms doivent être séparés par des points-virgules (exemple: Hari;Hari Seldon)
"""


def generate_submission():
    """
    Generates a submission file from the texts in the data/kaggle directory.
    """

    books = [
        (list(range(1, 20)), "paf"),
        (list(range(1, 19)), "lca"),
    ]

    df_dict = {"ID": [], "graphml": []}

    for chapters, book_code in tqdm(books):
        for chapter in tqdm(chapters):
            if book_code == "paf":
                path = f"data/kaggle/prelude_a_fondation/chapter_{chapter}.txt.preprocessed"
            else:
                path = f"data/kaggle/les_cavernes_d_acier/chapter_{chapter}.txt.preprocessed"
            print("Processing : ", path)
            graph_manager = GraphManager()
            experiment_name = os.path.join("save", "kaggle", book_code, "baseline")
            save_path = os.path.join(experiment_name, f"chapter_{chapter}.json")
            print("save_path : ", save_path)
            logger = JSONLogger(save_path)
            coocurrences = get_cooccurences_with_aliases_and_gpt(path, logger)
            graph_manager.add_cooccurrences(coocurrences)
            df_dict["ID"].append(f"{book_code}{chapter-1}")
            df_dict["graphml"].append("".join(html.unescape(s) if isinstance(s, str) else s for s in graph_manager.generate_graph()))

    df = pd.DataFrame(df_dict)
    df.set_index("ID", inplace=True)
    df.to_csv("submission.csv")


if __name__ == "__main__":
    generate_submission()
