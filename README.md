# Fondation GAN Graph

## Description

Ce projet fournit des outils pour générer un graphe de liens entre personnages à partir d'un texte. Il utilise un modèle de reconnaissance d'entités nommées (NER) pour identifier les personnages dans un texte et résoudre leurs alias à l'aide de règles et de fuzzy matching. Une classe GraphManager est également fournie pour créer des graphes à partir de listes de cooccurrences.

## Installation
```bash
pip install -r requirements.txt
pip install -e .
```

## Utilisation
Voici un exemple d'utilisation :
```python
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
```

Le code est disponible dans le fichier `examples/baseline.py`

## Fonctionnalités

- Génération de graphes de liens entre personnages à partir de textes.
- Résolution d'alias de personnages utilisant un modèle de NER et des techniques de fuzzy matching.
- Création et gestion de graphes avec la classe `GraphManager`.

## Structure du projet

Le projet est structuré de la manière suivante :

- `vroom` : contient le code source du projet.
    - `baseline.py` : contient les méthodes utilisées pour la baseline.
    - `GraphManager.py` : contient la classe `GraphManager` pour la gestion des graphes.
    - `NER.py` : contient les méthodes utilisées pour la *NER*.
    - `utils.py` : contient des fonctions utilitaires.
    - `alias.py` : contient les méthodes utilisées pour la résolution d'alias.
    - `metrics.py` : contient les méthodes utilisées pour calculer les métriques.
    - `loggers/` : contient des classes de `loggers` pour récupérer les ouptuts des différentes méthodes à chaque étape d'une pipeline.
- `data` : contient les données des livres et des chapitres pour *Kaggle*.
    - `books` : contient les livres au format `.txt` et `pdf`.
    - `finetuning_data` : contient les données pour le fine-tuning du modèle de *NER*.
    - `kaggle` : contient les chapitres au format `.txt`.
    - `output` : contient les données annotées par *ChatGPT* en mode *NER*.
    - `test_set` : contient des données annotées par nous même pour valider nos pipelines.
- `examples` : contient des exemples d'utilisation du projet.
- `tests` : contient les tests unitaires du projet.
- `notebooks` : contient les notebooks utilisés pour des expérimentations.
- `save` : contient les données pour chaque étape des pipelines d'annotation avec ChatGPT pour les chapitres de *Kaggle*.
- `scripts` : contient des scripts, permettant de diverses choses, allant du comptage de tokens au finetuning du modèle de *NER* en passant par une baseline complète avec *ChatGPT*.
- `submissions` : contient des soumissions pour le *Kaggle*.
## Tests

Pour lancer les tests unitaires, exécutez la commande suivante :
```bash
python -m unittest discover -s tests
```

## Contributeurs

- Adel MOUMEN
- Nicolas BATAILLE
- Gabriel DESBOUIS