from vroom.NER import tag_file, read_file
from vroom.metrics import evaluate

unlabeled_chapter = "/Users/adel-moumen/Documents/master/semester_3/application_innovation/Fondation-NER-Graph/data/test_set/prelude_a_fondation/chapter_1.gpt.labeled"
labeled_chapter = "/Users/adel-moumen/Documents/master/semester_3/application_innovation/Fondation-NER-Graph/data/test_set/prelude_a_fondation/chapter_1.labeled"

def extract_entities(text: str):
    """ Read a file and extract the entities from it.

    Args:
        file_path (str): The path to the file.

    Returns:
        list: A list of list of entities. Each list of entities represents an entity between <PER>word</PER>.
    """
    is_entity = False
    entities = []
    for word in text.split():
        if "<PER>" == word:
            entities.append({})
            entities[-1]["words"] = []
            is_entity = True
            continue

        if "</PER>" == word:
            is_entity = False
            continue

        if is_entity:
            entities[-1]["words"].append(word)
    return entities

def annotate_text_with_entities(text: str, entities: list):
    """ Annotate a text with entities.

    Args:
        text (str): The text to annotate.
        entities (list): A list of entities.

    Returns:
        str: The annotated text.
    """
    annotated_text = ""
    
    unfold_entities = []
    for entity in entities:
        wrd = ""
        for i, word in enumerate(entity["words"]):
            if i == 0:
                wrd += word
            else:
                wrd += " " + word
        unfold_entities.append(wrd)
    entities = unfold_entities
    print(entities)
    exit()
    for word in text.split():
        if word in entities:
            annotated_text += f"<PER> {word} </PER> "
        else:
            annotated_text += f"{word} "
    print(annotated_text)
    exit()
    return annotated_text

def annotate_entities(text, entities):
    annotated_text = []
    words = text.split()

    i = 0
    while i < len(words):
        # Removing punctuation to match entities
        word_cleaned = words[i].strip('.,;:"\'!?()[]{}')

        # Check if the current word is part of a multi-word entity
        multi_word_entity = None
        for entity in entities:
            if word_cleaned == entity.split()[0]:
                entity_words = entity.split()
                if words[i:i+len(entity_words)] == entity_words:
                    multi_word_entity = entity
                    break

        if multi_word_entity:
            annotated_text.append('<PER> ' + ' '.join(entity_words) + ' </PER> ')
            i += len(entity_words)
        else:
            annotated_text.append(words[i])
            i += 1

    return ' '.join(annotated_text)

gpt = read_file(unlabeled_chapter)
print('*' * 100)
true_label = read_file(labeled_chapter)

entities = extract_entities(gpt)

unlabeled_chapter = "/Users/adel-moumen/Documents/master/semester_3/application_innovation/Fondation-NER-Graph/data/test_set/prelude_a_fondation/chapter_1.unlabeled"
raw_text = read_file(unlabeled_chapter)

entities = [" ".join(entity["words"]) for entity in entities]
# remove empty entities
entities = [entity for entity in entities if entity != ""]

gpt = annotate_entities(raw_text, entities)
print(gpt[:600])
exit()
print(evaluate(gpt, true_label))
exit()

print(evaluate(tag_file(unlabeled_chapter), read_file(labeled_chapter)))