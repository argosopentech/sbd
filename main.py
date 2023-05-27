from typing import List

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
class SpacySentencizer(ISentenceBoundaryDetectionModel):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("sentencizer")

    def split_sentences(self, text: str, lang_code: str) -> List[str]:
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]

    def __str__(self):
        return "Spacy Sentencizer en_core_web_sm"


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
        return "Spacy Sentencizer en_core_web_sm"


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
    SpacySentencizer(),
    SpacySentencizerSmall(),
    ArgosTranslateSBD(),
    StanzaSBD(),
]
data = [
    "The black-and-yellow broadbill (Eurylaimus ochromalus) is a species of bird in Eurylaimidae, the typical broadbill family. It is small, with a black head, breastband, and upperparts, a white neckband, yellow streaking on the back and wings, and wine-pink underparts that turn yellow towards the belly. The beak is bright blue, with a green tip to the upper mandible and black edges. The black breastband is incomplete in females. The black-and-yellow broadbill occurs in Brunei, Indonesia, Malaysia, Myanmar, Singapore, and Thailand, in lowland forests up to an elevation of 1,220 m (4,000 ft). It is mainly insectivorous, but also eats molluscs and some fruit. It breeds during the dry season with both sexes helping build a large, untidy nest from moss, fungal mycelia, and leaves. The clutch is usually 2–3 eggs, and sometimes includes a fourth runt egg. They are incubated by both sexes. The species is listed as near-threatened by the IUCN due to a decline in its population caused by habitat loss. (Full article...)"
]

for model in sbd_models:
    print(f"Model: {model}")
    for text in data:
        print(f"Input: {text}")
        print(f"Output: {model.split_sentences(text, 'en')}")
        print()
    print()
