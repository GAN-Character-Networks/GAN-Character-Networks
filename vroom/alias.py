r"""  Package for managing aliases in NER tasks.

Authors
--------
 * Nicolas Bataille 2023
"""

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
    group_index = 0
    for entity in entities:
        name = entity["word"].lower().replace(" ", "_")
        if name not in group_associations:
            best_score = 0
            best_name = ""
            for other_entity in entities:
                other_name = other_entity["word"].lower().replace(" ", "_")
                # print(f"Comparing {name} with {other_name}")
                if other_name != name:
                    # print("Not the same name")
                    score = fuzz.partial_token_sort_ratio(name, other_name)
                    if score > best_score and score >= treshold:
                        print(
                            f"Best score for {name} is {score} with {other_name}"
                        )
                        best_score = score
                        best_name = other_name
            if best_score > 0:
                # print(f"The best name for {name} is {best_name}")
                if best_name not in group_associations:
                    group_associations[best_name] = len(group_associations)
                group_associations[name] = group_associations[best_name]
            else:
                # print(f"{name} has no alias")
                group_associations[name] = group_index
                group_index += 1

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
