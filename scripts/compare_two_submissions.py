r""" Compare two submissions and ask the user to choose the best one for each chapter.

Usage:
    python compare_two_submissions.py
    
    This script will display the nodes and edges of the graph for each chapter of the two submissions.
    Then, it will ask the user to choose the best submission for each chapter.
    Finally, it will save the user choices in a JSON file.
    
    The two submissions must be in CSV format and have the following structure:
    chapter_id,graphml
    paf0,"<graphml>...</graphml>"
    paf1,"<graphml>...</graphml>"
    ...
    
Authors
-------
 * Gabriel Desbouis 2024
"""

import csv
import json
import xml.etree.ElementTree as ET


def parse_graphml(data):
    """
    Parse a graphml file and return the nodes and edges.
    
    Args:
        data (str): The content of the graphml file.
        
    Returns:
            nodes (dict): A dictionary of nodes with the following structure: {id: name}
            edges (list): A list of edges with the following structure: [(source, target)]
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


def display_graph_info(filename, chapter, nodes, edges):
    """
    Display the nodes and edges of a graph.

    Args:
        filename (str): The name of the file.
        chapter (str): The chapter of the graph.
        nodes (dict): The nodes of the graph.
        edges (list): The edges of the graph.
    """
    print(f"Fichier {filename}, Chapitre: {chapter}")
    for id, name in nodes.items():
        print(f"{id}: {name}")
    print("\nLiens entre les entités de ce chapitre:")
    for edge in edges:
        print(f"{edge[0]} -> {edge[1]}")
    print("\n")


def read_csv_and_display_info(filename, chapter):
    """
    Read a CSV file and display the nodes and edges of the graph for a given chapter.

    Args:
        filename (str): The name of the file.
        chapter (str): The chapter of the graph.
    """
    with open(filename, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for row in reader:
            if row[0] == chapter:
                nodes, edges = parse_graphml(row[1])
                display_graph_info(filename, chapter, nodes, edges)


def main():
    books = [
        (list(range(0, 19)), "paf"),
        (list(range(0, 18)), "lca"),
    ]
    user_choices = {}

    for book in books:
        chapters, prefix = book
        for chapter_number in chapters:
            chapter = f"{prefix}{chapter_number}"
            file1 = "../submission_058.csv"  # Remplacez par le chemin d'accès au premier fichier CSV
            file2 = "../submission_070.csv"  # Remplacez par le chemin d'accès au second fichier CSV

            read_csv_and_display_info(file1, chapter)
            read_csv_and_display_info(file2, chapter)

            choice = input(
                f"Préférez-vous les données du fichier 1 ou du fichier 2 pour le chapitre {chapter}? "
            )
            user_choices[chapter] = choice

    # Sauvegarde des choix dans un fichier JSON
    with open("user_choices.json", "w") as json_file:
        json.dump(user_choices, json_file)


if __name__ == "__main__":
    main()
