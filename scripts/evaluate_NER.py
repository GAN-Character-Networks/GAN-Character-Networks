from vroom.NER import tag_file, read_file
from vroom.metrics import evaluate

unlabeled_chapter = "/Users/adel-moumen/Documents/master/semester_3/application_innovation/Fondation-NER-Graph/data/test_set/prelude_a_fondation/chapter_1.unlabeled"
labeled_chapter = "/Users/adel-moumen/Documents/master/semester_3/application_innovation/Fondation-NER-Graph/data/test_set/prelude_a_fondation/chapter_1.labeled"

print(evaluate(tag_file(unlabeled_chapter), read_file(labeled_chapter)))