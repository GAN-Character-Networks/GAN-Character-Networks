from vroom.NER import *
from vroom.metrics import evaluate
import os

unlabeled_chapter = os.path.join("data", "test_set", "prelude_a_fondation", "chapter_1.unlabeled")
labeled_chapter = os.path.join("data", "test_set", "prelude_a_fondation", "chapter_1.labeled")

entities, chunks = get_entities_from_file(unlabeled_chapter, device="cuda")

all_entities_names = []
for chunk in entities: 
    all_entities_names += [
        entity["word"] for entity in chunk
    ]
entities = set(all_entities_names)

unlabeled_text_tagged = tag_text_with_entities(unlabeled_chapter, entities)

labeled_text_tagged = read_file(labeled_chapter)

import re


def separate_words(text):
    # Use regular expression to find words and punctuation, including special words like <PER> and </PERS>
    words = re.findall(r'\b\w+\b|[.,;!?<>/]+|<PER>|</PER>', text)
    return words

def merge_special_words(word_list):
    merged_list = []
    current_word = ""

    for word in word_list:
        if word in ['<', 'PER', '</', '>']:
            current_word += word
        else:
            if current_word:
                merged_list.append(current_word)
                current_word = ""
            merged_list.append(word)

    if current_word:
        merged_list.append(current_word)

    return merged_list

def get_positions_of_entities(text):
    words = separate_words(text)
    words = merge_special_words(words)
    positions = {}
    current_entity = []
    current_entity_start = 0
    current_entity_end = 0
    i = 0
    for word in words:
        if word == "<PER>":
            current_entity = []
            current_entity_start = i
        elif word == "</PER>":
            current_entity_end = i
            positions[(current_entity_start, current_entity_end)] = ' '.join(current_entity)
        else:
            i += 1
            current_entity.append(word)
    return positions

positions_ner = get_positions_of_entities(unlabeled_text_tagged)
true_position = get_positions_of_entities(labeled_text_tagged)

def evaluate_ner(positions_ner, true_position):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # check if the position of the entity is the same
    for entity in positions_ner:
        if entity in true_position:
            true_positives += 1
        else:
            false_positives += 1
        
    # check if the entity is missing
    for entity in true_position:
        if entity not in positions_ner:
            false_negatives += 1

    print("true_positive", true_positives)
    print("false_positive", false_positives)
    print("false_negative", false_negatives)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = true_positives / (true_positives + false_positives + false_negatives)

    print("precision", precision)
    print("recall", recall)
    print("f1_score", f1_score)
    print("accuracy", accuracy)

evaluate_ner(positions_ner, true_position)
