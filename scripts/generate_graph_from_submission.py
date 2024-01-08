r""" Generate a graph from a submission file.
Usage:
    python generate_graph_from_submission.py
    
    This script will generate a graph for each chapter of the submission file.
    The submission file must be in CSV format and have the following structure:
    chapter_id,graphml
    paf0,"<graphml>...</graphml>"
    paf1,"<graphml>...</graphml>"
    ...

Authors
-------
    * Gabriel Desbouis 2024
"""

import csv
import xml.etree.ElementTree as ET
from vroom.GraphManager import GraphManager
import os


def parse_graphml(data):
    """
    Parse a graphml file and return the nodes and edges.
    """
    root = ET.fromstring(data)
    nodes = {}
    edges = []
    for node in root.findall(".//{http://graphml.graphdrawing.org/xmlns}node"):
        id = node.get("id")
        name = node.find(".//{http://graphml.graphdrawing.org/xmlns}data").text
        nodes[id] = name
    for edge in root.findall(".//{http://graphml.graphdrawing.org/xmlns}edge"):
        source = edge.get("source")
        target = edge.get("target")
        edges.append((source, target))
    return nodes, edges


def generate_cooccurrences(nodes):
    """
    Generate cooccurrences from a list of nodes.
    """
    cooccurrences = []
    for id1, aliases1 in nodes.items():
        for id2, aliases2 in nodes.items():
            if id1 != id2:
                cooccurrences.append(([aliases1], [aliases2]))
    return cooccurrences


def process_submission_file(filename):
    """
    Generate a graph for each chapter of the submission file.
    """
    with open(filename, newline="", encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        next(reader)
        for row in reader:
            chapter_id, graphml_data = row
            print(f"Processing chapter {chapter_id}")
            print(graphml_data)
            nodes, _ = parse_graphml(graphml_data)
            cooccurrences = generate_cooccurrences(nodes)
            
            graphManager = GraphManager()
            graphManager.add_cooccurrences(cooccurrences)
            graphManager.save_graph_to_graphml(f"../data/kaggle/output_graphs/{chapter_id}.graphml")


def main():
    submission_file = "../submissions/filtered_output.csv"  # Chemin du fichier CSV de soumission

    # Créer le dossier de sortie si nécessaire
    if not os.path.exists("../data/kaggle/output_graphs"):
        os.makedirs("../data/kaggle/output_graphs")

    process_submission_file(submission_file)


if __name__ == "__main__":
    main()
