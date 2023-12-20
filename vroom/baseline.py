r"""  Pipeline for cooccurences extraction with baseline methods

Authors
--------
 * Nicolas Bataille 2023
"""
from vroom.NER import get_entities_from_file
from vroom.alias import get_aliases_fuzzy
from vroom.cooccurences import get_cooccurences


def get_cooccurences_from_text(path: str):
    """
    Get the coocurences of characters from the given text.

    Args:
        path (str): The path of the text file.

    Returns:
        list: A list of tuples representing the interactions between entities in the text.
    """

    entities, chunks = get_entities_from_file(path)
    return get_cooccurences(chunks, entities)


def get_cooccurences_with_aliases(path: str):
    """
    Get the aliases of the cooccurences of characters from the given text.

    Args:
        path (str): The path of the text file.

    Returns:
        list: A list of tuples representing the interactions between entities in the text.
    """
    cooccurences = get_cooccurences_from_text(path)
    entities, _ = get_entities_from_file(path)
    entities = [entity for sublist in entities for entity in sublist]
    aliases = get_aliases_fuzzy(entities, 99)

    for cooc in cooccurences:
        no_alias_1 = True
        no_alias_2 = True
        for alias in aliases:
            if cooc[0] in alias:
                no_alias_1 = False
            if cooc[1] in alias:
                no_alias_2 = False
        if no_alias_1:
            print("no alias 1 : ", cooc[0])
        if no_alias_2:
            print("no alias 2 : ", cooc[1])

    print("aliases : ", aliases)

    cooccurences_aliases = []
    for cooccurence in cooccurences:
        cooc_1_aliases = [
            alias
            for alias in aliases
            if cooccurence[0].lower() in [a.lower() for a in alias]
        ][0]
        cooc_2_aliases = [
            alias
            for alias in aliases
            if cooccurence[1].lower() in [a.lower() for a in alias]
        ][0]
        cooccurences_aliases.append((cooc_1_aliases, cooc_2_aliases))

    return cooccurences_aliases


# TODO: Idée de solving d'alias : Récup tout les alias de tout les chapitres du livre, puis construire par chapitre la liste d'alias correspondante
