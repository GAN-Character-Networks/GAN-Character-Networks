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
    cooccurences_aliases = []
    for cooccurence in cooccurences:
        cooc_1_aliases = [
            alias for alias in aliases if cooccurence[0] in alias
        ][0]
        cooc_2_aliases = [
            alias for alias in aliases if cooccurence[1] in alias
        ][0]
        cooccurences_aliases.append((cooc_1_aliases, cooc_2_aliases))

    return cooccurences_aliases
