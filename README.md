# SBD
Sentence boundary detection for machine translation

## Benchmarks
| Model                  | Average Accuracy  | Average Runtime (seconds) |
| ---------------------- | ----------------- | ------------------------- |
| Spacy en_core_web_sm   | 0.924311498164287 | 0.0250468651453654        |
| Spacy xx_sent_ud_sm    | 0.924311498164287 | 0.00476229190826416       |
| Argos Translate 2 Beta | 0.515548280365557 | 1.87798078854879          |
| Stanza en              | 0.924311498164287 | 0.0219400326410929        |

#### Libraries tested
- [Argos Translate 2 beta](https://github.com/argosopentech/argos-translate/tree/v2)
- [Stanza](https://stanfordnlp.github.io/stanza/)
- [Spacy](https://spacy.io/)

#### Data
The data is from Wikipedia and I (native English speaker) manually split the sentences. Currently only English data is used but this script can easily be extended for more languages.

https://en.wikipedia.org/
