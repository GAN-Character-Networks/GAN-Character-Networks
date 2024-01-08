r""" This script allows to generate a submission file for the Kaggle competition.

Authors
-------
 * Gabriel DESBOUIS 2023
"""


from vroom.baseline import get_cooccurences_with_aliases_and_gpt
from vroom.GraphManager import GraphManager
import pandas as pd
from tqdm import tqdm
from vroom.loggers import JSONLogger
import html
import os

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
