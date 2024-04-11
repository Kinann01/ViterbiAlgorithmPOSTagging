# Viterbi Algorithm

## Overview
This project contains a Python script that implements a Part-Of-Speech (POS) tagger using the Viterbi algorithm. The tagger is trained on a given dataset to learn tagging patterns and can then predict POS tags for new text.

## Requirements
- Python 3
- numpy

### Training the Tagger
To train the tagger, you'll need a file containing the training data. The training data should be formatted with one word and its corresponding tag per line, separated by a tab. A blank line is used to separate sentences.

### Evaluating the Tagger
To evaluate the tagger, you'll need a file containing the test data. Open the src/main.py file and replace the 'tagger-eval.tsv' with the path to your test data file.

### Usage
Run the following command in your terminal:

```sh
python src/main.py path_to_training_data
```