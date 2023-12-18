r"""  Package for managing aliases in NER tasks.

Authors
--------
 * Nicolas Bataille 2023
"""

# import numpy as np
# from sklearn.cluster import DBSCAN
from textdistance import jaro_winkler
from rapidfuzz import fuzz


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


def get_aliases_jaro_winkler(entities: list, treshold: float = 0.8):
    """
    Returns a list of aliases based on the Jaro-Winkler distance.

    Args:
        entities (list): A list of dictionaries representing the entities.
        treshold (int): The treshold for the Jaro-Winkler distance.

    Returns:
        list: A list of dictionaries representing the aliases.
    """
    aliases = []
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
            # print(f"The best name for {name} is {best_name}")
            if best_name not in group_associations:
                group_associations[best_name] = len(group_associations)
            group_associations[name] = group_associations[best_name]
        else:
            # print(f"{name} has no alias")
            group_associations[name] = len(group_associations)

    for entity in group_associations:
        if group_associations[entity] == -1:
            continue
        group = []
        for other_entity in group_associations:
            if (
                group_associations[other_entity] == group_associations[entity]
                and other_entity != entity
                and group_associations[other_entity] != -1
            ):
                group.append(other_entity)
                group_associations[other_entity] = -1
        group.append(entity)
        aliases.append(group)
        group_associations[entity] = -1

    return aliases


def get_best_alias_fuzzy(
    name: str,
    entities: list,
    group_associations: list,
    treshold: int = 80,
    alias_list_only: bool = False,
):
    """
    Returns the best alias and its score based on the fuzzy algorithm.

    Args:
        name (str): The name of the entity.
        entities (list): A list of dictionaries representing the entities.
        group_associations (list): A list of already defined entities and alias ids.
        treshold (int): The treshold for the fuzzy score.
        alias_list_only (bool): If True, only check in the group_associations list.

    Returns:
        str: The best alias.
        float: The score of the best alias.
    """
    best_score = 0
    best_name = ""
    # Look first in the already found aliases
    for entity in group_associations:
        if entity != name:
            score = fuzz.partial_token_sort_ratio(name, entity)
            if score > best_score and score >= treshold:
                best_score = score
                best_name = entity
    # Then look over every entities
    if not alias_list_only:
        for other_entity in entities:
            other_name = other_entity["word"].lower().replace(" ", "_")
            # print(f"Comparing {name} with {other_name}")
            if other_name != name:
                # print("Not the same name")
                score = fuzz.partial_token_sort_ratio(name, other_name)
                if score > best_score and score >= treshold:
                    best_score = score
                    best_name = other_name

    return best_name, best_score


def get_aliases_fuzzy(entities: list, treshold: int = 80):
    """
    Returns a list of aliases based on the fuzzy algorithm.

    Args:
        entities (list): A list of dictionaries representing the entities.
        treshold (int): The treshold for the fuzzy score.

    Returns:
        list: A list of dictionaries representing the aliases.
    """
    aliases = []
    group_associations = {}
    raw_names = {}
    for entity in entities:
        raw_names[entity["word"].lower().replace(" ", "_")] = entity["word"]
        name = entity["word"].lower().replace(" ", "_")
        # print(f"Looking for alias for {name}")
        if group_associations.get(name) is None:

            best_name, best_score = get_best_alias_fuzzy(
                name, entities, group_associations, treshold
            )

            if best_score > treshold:
                # print(f"Best score for {name} is {best_score} with {best_name}")
                if best_name not in group_associations:

                    best_name_alias, _ = get_best_alias_fuzzy(
                        best_name,
                        entities,
                        group_associations,
                        treshold,
                        alias_list_only=True,
                    )
                    # print(
                    # f"Best alias for best name {best_name} is {best_name_alias} with {best_score}"
                    # )
                    if best_name_alias == name:
                        group_associations[best_name] = len(group_associations)
                    elif best_name_alias != "":
                        group_associations[best_name] = group_associations[
                            best_name_alias
                        ]
                    else:
                        group_associations[best_name] = len(group_associations)
                group_associations[name] = group_associations[best_name]
            else:
                # print(f"{name} has no alias")
                group_associations[name] = len(group_associations)
        # else:
        # print("================== Already has an alias ==================")
        # print(group_associations)

    for entity in group_associations:
        if group_associations[entity] == -1:
            continue
        group = []
        for other_entity in group_associations:
            if (
                group_associations[other_entity] == group_associations[entity]
                and other_entity != entity
                and group_associations[other_entity] != -1
            ):
                group.append(raw_names[other_entity])
                group_associations[other_entity] = -1
        group.append(raw_names[entity])
        aliases.append(group)
        group_associations[entity] = -1

    # TODO: refaire une passe derrière pour voir les alias qui seraient
    # potentiellement intéressants de mergent ensemble (exemple de daneel pour le chapitre 2)

    return aliases
