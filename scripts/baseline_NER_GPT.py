import re
from vroom.NER import *
import os

unlabeled_chapter = os.path.join("data", "test_set", "prelude_a_fondation", "chapter_1.unlabeled")

entities, chunks = get_entities_from_file(unlabeled_chapter)

all_entities_names = []
for chunk in entities: 
    all_entities_names += [
        entity["word"] for entity in chunk
    ]
entities = set(all_entities_names)

text_tagged = tag_text_with_entities(unlabeled_chapter, entities)

print(text_tagged)