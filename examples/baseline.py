from vroom.baseline import get_cooccurences_from_text
from vroom.GraphManager import GraphManager


def baseline(text):
    return get_cooccurences_from_text(text)


if __name__ == "__main__":
    path = "../data/kaggle/les_cavernes_d_acier/chapter_1.txt.preprocessed"

    graphManager = GraphManager()
    coocurrences = baseline(path)
    graphManager.add_cooccurrences(coocurrences)
    graphManager.save_graph_to_graphml("chapter_1.graphml")
