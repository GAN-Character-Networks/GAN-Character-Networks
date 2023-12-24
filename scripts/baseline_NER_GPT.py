import re
from vroom.NER import *
import os

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

unlabeled_chapter = os.path.join("data", "test_set", "prelude_a_fondation", "chapter_1.unlabeled")

entities, chunks = get_entities_from_file(unlabeled_chapter, device="cuda")

all_entities_names = []
for chunk in entities: 
    all_entities_names += [
        entity["word"] for entity in chunk
    ]
entities = set(all_entities_names)

text_tagged = tag_text_with_entities(unlabeled_chapter, entities)

print(text_tagged)