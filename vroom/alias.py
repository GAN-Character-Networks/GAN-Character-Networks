r"""  Package for managing aliases in NER tasks.

Authors
--------
 * Nicolas Bataille 2023
"""

from textdistance import jaro_winkler


def remove_duplicates_by_word(data):
    """
    Removes duplicate entities based on the 'word' attribute.

    Args:
        data (list): A list of dictionaries representing the entities.

    Returns:
        list: A list of dictionaries with duplicate entities removed.
    """
    unique_dicts = []
    words = set()

    for entity in data:
        word = entity["word"]
        if word not in words:
            unique_dicts.append(entity)
            words.add(word)

    return unique_dicts


def get_aliases_jaro_winkler(entities: list, treshold: int = 0.8):
    """
    Returns a list of aliases based on the Jaro-Winkler distance.

    Args:
        entities (list): A list of dictionaries representing the entities.
        treshold (int): The treshold for the Jaro-Winkler distance.

    Returns:
        list: A list of dictionaries representing the aliases.
    """
    # aliases = []
    group_associations = {}
    for entity in entities:
        name = entity["word"].lower().replace(" ", "_")
        best_score = 0
        best_name = ""
        for other_entity in entities:
            other_name = other_entity["word"].lower().replace(" ", "_")
            # print(f"Comparing {name} with {other_name}")
            if other_name != name:
                # print("Not the same name")
                score = jaro_winkler(name, other_name)
                if score > best_score and score >= treshold:
                    best_score = score
                    best_name = other_name
        if best_score > 0:
            print(f"The best name for {name} is {best_name}")
            if best_name not in group_associations:
                group_associations[best_name] = len(group_associations)
            group_associations[name] = group_associations[best_name]
        else:
            print(f"{name} has no alias")
            group_associations[name] = len(group_associations)

    print(group_associations)


"""
    gérer les alias avec les scores de jaro winkler :

    - Pour chaque mots, on regarde le mot avec lequel il a le meilleur score
    - Ensuite, on regarde si le meilleur mot trouvé n'a pas déjà été passé en comparaison
        - Si non, alors on ajoute le meilleur mot à un dict de groupe, et on y associe un id de groupe (donc un nouveau cluster)
        - Si oui, alors on ajoute le que l'on compare au groupe du meilleur mot trouvé
    - On fait ça pour tous les mots
    - On return une liste de liste d'alias

"""
