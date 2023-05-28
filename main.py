from typing import List
import csv
import json
from difflib import SequenceMatcher
import time
from collections import defaultdict

import argostranslate.translate
import spacy
import stanza


class ISentenceBoundaryDetectionModel:
    def split_sentences(self, text: str, lang_code: str) -> List[str]:
        raise NotImplementedError


# Use spacy to split sentences
class SpacySentenceBoundaryDetectionModel(ISentenceBoundaryDetectionModel):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def split_sentences(self, text: str, lang_code: str) -> List[str]:
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]

    def __str__(self):
        return "Spacy en_core_web_sm"


# Spacy sentence boundary detection Sentencizer
# https://community.libretranslate.com/t/sentence-boundary-detection-for-machine-translation/606/3
class SpacySentencizerSmall(ISentenceBoundaryDetectionModel):
    def __init__(self):
        self.nlp = spacy.load("xx_sent_ud_sm")
        self.nlp.add_pipe("sentencizer")

    def split_sentences(self, text: str, lang_code: str) -> List[str]:
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]

    def __str__(self):
        return "Spacy xx_sent_ud_sm"


class ArgosTranslateSBD(ISentenceBoundaryDetectionModel):
    def split_sentences(self, text: str, lang_code: str) -> List[str]:
        return argostranslate.translate.chunk(text, lang_code)

    def __str__(self):
        return "Argos Translate 2 Beta"


class StanzaSBD(ISentenceBoundaryDetectionModel):
    def __init__(self):
        self.split_sentences("Hello World", "en")

    def split_sentences(self, text: str, lang_code: str) -> List[str]:
        self.stanza_pipeline = stanza.Pipeline(lang_code, processors="tokenize")
        stanza_sbd = self.stanza_pipeline(text)
        return [sentence.text for sentence in stanza_sbd.sentences]

    def __str__(self):
        return "Stanza en"


sbd_models = [
    SpacySentenceBoundaryDetectionModel(),
    SpacySentencizerSmall(),
    ArgosTranslateSBD(),
    StanzaSBD(),
]


# Get testing data
# Testing data in sbd-testing-data.jsonl
# Example line:
# {"lang_code": "en", "q": "This is test data. This is another sentence.", "split_sentences": ["This is test data. ", "This is another sentence."]]}

TEST_DATA_FILE = "sbd-testing-data.jsonl"

data = []
with open(TEST_DATA_FILE, "r") as f:
    for line in f:
        data.append(json.loads(line))
        
# Test models
class Experiment:
    def __init__(self, model_name: str, similarity: float, run_time: float):
        self.model_name = model_name
        self.similarity = similarity
        self.run_time = run_time

    def __str__(self) -> str:
        return f"{self.model_name}: similarity: {self.similarity} runtime: {self.run_time}"

def similarity_of_str_lst(expected: List[str], actual: List[str]) -> float:
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    unmatched_expected = expected.copy()
    similarity_sum = 0.0
    for actual_sentence in actual:
        # Find best matching expected sentence
        best_match = None
        best_match_ratio = 0
        for expected_sentence in unmatched_expected:
            ratio = SequenceMatcher(None, expected_sentence, actual_sentence).ratio()
            if ratio > best_match_ratio:
                best_match = expected_sentence
                best_match_ratio = ratio
        # Remove best match from unmatched expected sentences
        if best_match is not None:
            unmatched_expected.remove(best_match)
        
        # Sum the similarities by sentence
        similarity_sum += best_match_ratio

    return similarity_sum / len(max(expected, actual, key=len))

experiments = defaultdict(list)

for model in sbd_models:
    print(f"Testing {model}")
    for test in data:
        expected = test["split_sentences"]
        start_time = time.time()
        actual = model.split_sentences(test["q"], test["lang_code"])
        end_time = time.time()
        run_time = end_time - start_time
        similarity = similarity_of_str_lst(expected, actual)
        experiment = Experiment(str(model), similarity, run_time)
        experiments[str(model)].append(experiment)

compiled_data = list()

data_headers = ["Model", "Average Accuracy", "Average Runtime (seconds)"]

for model, experiments in experiments.items():
    average_accuracy = sum([experiment.similarity for experiment in experiments]) / len(experiments)
    average_runtime = sum([experiment.run_time for experiment in experiments]) / len(experiments)
    compiled_data.append([model, average_accuracy, average_runtime])
    
# Write to csv
with open("sbd-testing-results.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(data_headers)
    writer.writerows(compiled_data)
print("Compiled results written to sbd-testing-results.csv")