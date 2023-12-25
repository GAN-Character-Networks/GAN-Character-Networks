"""This file contains the metrics used to evaluate the performance of the
system.

It includes various metrics to evaluate the NER to the coocurrences algorithm.

Authors
--------
 * Adel Moumen 2023
"""


def get_entities_from_file(text: str):
    """ Read a file and extract the entities from it.

    Args:
        file_path (str): The path to the file.

    Returns:
        list: A list of list of entities. Each list of entities represents an entity between <PER>word</PER>.
    """
    is_entity = False
    entities = []
    start_pos = 0
    end_pos = 0
    for word in text.split():
        if "<PER>" == word:
            entities.append({})
            entities[-1]["words"] = []
            entities[-1]["start_pos"] = start_pos
            is_entity = True
            continue

        if "</PER>" == word:
            entities[-1]["end_pos"] = end_pos
            is_entity = False
            continue

        if is_entity:
            entities[-1]["words"].append(word)

        start_pos += len(word) + 1
        end_pos += len(word) + 1

    return entities


def evaluate(predicted_text: str, true_text: str):
    """ This function evaluates the performance of the NER system on
    different metrics such as precision, recall, f1 score, and accuracy.

    Args:
        predicted_text (str): The path to the file containing the predicted text.
        true_text (str): The path to the file containing the true text.

    Returns:
        dict: A dictionary containing the metrics.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    predicted_entities = get_entities_from_file(predicted_text)
    true_entities = get_entities_from_file(true_text)

    print("predicted_entities", predicted_entities)
    print("*" * 100)
    print("true_entities", true_entities)

    for predicted_entity in predicted_entities:
        for true_entity in true_entities:
            if predicted_entity["words"] == true_entity["words"]:
                if (
                    predicted_entity["start_pos"] == true_entity["start_pos"]
                    and predicted_entity["end_pos"] == true_entity["end_pos"]
                ):
                    true_positives += 1
                    break
        else:
            false_positives += 1

    for true_entity in true_entities:
        for predicted_entity in predicted_entities:
            if predicted_entity["words"] == true_entity["words"]:
                if (
                    predicted_entity["start_pos"] == true_entity["start_pos"]
                    and predicted_entity["end_pos"] == true_entity["end_pos"]
                ):
                    break
        else:
            false_negatives += 1
    print("true_positives", true_positives)
    print("false_positives", false_positives)
    print("false_negatives", false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = true_positives / (
        true_positives + false_positives + false_negatives
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }
