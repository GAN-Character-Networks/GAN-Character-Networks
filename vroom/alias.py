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
    name: str, entities: list, group_associations: list, treshold: int = 80
):
    """
    Returns the best alias and its score based on the fuzzy algorithm.

    Args:
        name (str): The name of the entity.
        entities (list): A list of dictionaries representing the entities.
        group_associations (list): A list of already defined entities and alias ids.
        treshold (int): The treshold for the fuzzy score.

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

            if best_score > 50:
                # print(f"Best score for {name} is {best_score} with {best_name}")
                if best_name not in group_associations:

                    best_name_alias, _ = get_best_alias_fuzzy(
                        best_name, entities, group_associations, treshold
                    )
                    # print(
                    # f"Best alias for best name {best_name} is {best_name_alias} with {best_score}"
                    # )
                    if best_name_alias == name:
                        group_associations[best_name] = len(group_associations)
                    else:
                        group_associations[best_name] = group_associations[
                            best_name_alias
                        ]

                    group_associations[best_name] = len(group_associations)
                group_associations[name] = group_associations[best_name]
            elif best_score <= 50:
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


# def get_best_alias_fuzzy(
#     name: str, entities: list, group_associations: list, treshold: int = 80
# ):
#     """
#     Returns the best alias and its score based on the fuzzy algorithm.

#     Args:
#         name (str): The name of the entity.
#         entities (list): A list of dictionaries representing the entities.
#         group_associations (list): A list of already defined entities and alias ids.
#         treshold (int): The treshold for the fuzzy score.

#     Returns:
#         str: The best alias.
#         float: The score of the best alias.
#     """
#     best_score = 0
#     best_name = ""
#     #look over every entities
#     for other_entity in entities:
#         other_name = other_entity["word"].lower().replace(" ", "_")
#         # print(f"Comparing {name} with {other_name}")
#         if other_name != name:
#             # print("Not the same name")
#             score = fuzz.partial_token_sort_ratio(name, other_name)
#             if score >= best_score and score >= treshold:
#                 best_score = score
#                 best_name = other_name
#     # Then look over every already existing aliases
#     for entity in group_associations:
#         if entity != name:
#             score = fuzz.partial_token_sort_ratio(name, entity)
#             if score >= best_score and score >= treshold:
#                 best_score = score
#                 best_name = entity
#     return best_name, best_score


# def validate_aliases(aliases: list, threshold: int = 95, min_samples: int = 1):
#     """
#     Validate the aliases associations and modify them if needed.

#     Args:
#         aliases (list): A list of dictionaries representing the aliases.

#     Returns:
#         list: A list of dictionaries representing the aliases.
#     """
#     # Calculer les similarités entre les alias
#     similarity_matrix = np.zeros((len(aliases), len(aliases)))
#     for i in range(len(aliases)):
#         for j in range(len(aliases)):
#             similarity_matrix[i, j] = fuzz.token_set_ratio(" ".join(aliases[i]), " ".join(aliases[j]))

#     # Regrouper les alias similaires en utilisant DBSCAN
#     dbscan = DBSCAN(eps=100-threshold, min_samples=min_samples, metric="precomputed")
#     clusters = dbscan.fit_predict(100 - similarity_matrix)

#     # Fusionner les groupes d'alias
#     merged_alias_groups = {}
#     for i, cluster_id in enumerate(clusters):
#         if cluster_id not in merged_alias_groups:
#             merged_alias_groups[cluster_id] = []
#         merged_alias_groups[cluster_id].extend(aliases[i])

#     return list(merged_alias_groups.values())


# def recursive_find_aliases(name: str, group_associations: dict, entities: list, treshold: int = 80):
#     best_name, best_score = get_best_alias_fuzzy(
#                 name, entities, group_associations, treshold
#             )

#     if best_score > treshold:
#         print(f"Best score for {name} is {best_score} with {best_name}")
#         if best_name not in group_associations:
#             group_associations = recursive_find_aliases(best_name, group_associations, entities, treshold)

#             best_name_alias, _ = get_best_alias_fuzzy(
#                 best_name, entities, group_associations, treshold
#             )
#             print(
#                 f"Best alias for best name {best_name} is {best_name_alias} with {best_score}"
#             )
#         if best_name_alias == name:
#             group_associations[best_name] = len(group_associations)
#             else:
#                 print(best_name_alias)
#                 print(best_name)
#                 group_associations[best_name] = group_associations[
#                     best_name_alias
#                 ]

#             group_associations[best_name] = len(group_associations)
#         group_associations[name] = group_associations[best_name]
#     else:
#         print(f"{name} has no alias")
#         group_associations[name] = len(group_associations)

#     return group_associations

# def get_aliases_fuzzy(entities: list, treshold: int = 80):
#     """
#     Returns a list of aliases based on the fuzzy algorithm.

#     Args:
#         entities (list): A list of dictionaries representing the entities.
#         treshold (int): The treshold for the fuzzy score.

#     Returns:
#         list: A list of dictionaries representing the aliases.
#     """
#     aliases = []
#     group_associations = {}
#     raw_names = {}
#     for entity in entities:
#         raw_names[entity["word"].lower().replace(" ", "_")] = entity["word"]
#         name = entity["word"].lower().replace(" ", "_")
#         print(f"Looking for alias for {name}")
#         if group_associations.get(name) is None:

#             group_associations = recursive_find_aliases(name, group_associations, entities, treshold)

#         else:
#             print("================== Already has an alias ==================")
#             print(group_associations)


#     #group_associations = validate_aliases(group_associations)

#     for entity in group_associations:
#         if group_associations[entity] == -1:
#             continue
#         group = []
#         for other_entity in group_associations:
#             if (
#                 group_associations[other_entity] == group_associations[entity]
#                 and other_entity != entity
#                 and group_associations[other_entity] != -1
#             ):
#                 group.append(raw_names[other_entity])
#                 group_associations[other_entity] = -1
#         group.append(raw_names[entity])
#         aliases.append(group)
#         group_associations[entity] = -1

#     # TODO: refaire une passe derrière pour voir les alias qui seraient
#     # potentiellement intéressants de mergent ensemble (exemple de daneel pour le chapitre 2)


#     return aliases
