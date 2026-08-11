# Naive Bayes From Scratch

This project is a from-scratch implementation of Naive Bayes using Python and NumPy. Naive Bayes is a probabilistic classification algorithm that predicts the class of a text by combining the prior probability of each class with the probability of the words appearing in that class.

The implementation is in [naive_bayes_classification.py](naive_bayes_classification.py).

## Project Overview

The script trains a simple text classifier on short movie-related sentences. Each sentence is labeled as:

- `positive`
- `negative`

The model learns which words are more common in positive and negative examples, then predicts the sentiment of new sentences.

## What the Script Covers

- Importing NumPy
- Tokenizing text by lowercasing and splitting words
- Creating a vocabulary from the training text
- Counting documents in each class
- Counting word frequencies for each class
- Calculating prior probabilities for each class
- Applying Laplace smoothing for unseen words
- Scoring each class for a new sentence
- Predicting the class with the highest score

## Maths Behind the Model

Naive Bayes uses Bayes' theorem:

```text
P(class | words) = P(class) * P(words | class) / P(words)
```

For classification, the denominator is the same for every class, so the model compares scores like this:

```text
score = P(class) * P(word1 | class) * P(word2 | class) * ... * P(wordN | class)
```

The prior probability is:

```text
P(class) = number of documents in class / total number of documents
```

The script uses Laplace smoothing when calculating word probabilities:

```text
P(word | class) = (word count in class + 1) / (total words in class + vocabulary size)
```

This prevents the probability from becoming zero when a word was not seen in a class during training.

## How to Run

1. Open the `Naive_Bayes` folder.
2. Run the script:

```bash
python naive_bayes_classification.py
```

3. Check the printed predictions for the test sentences.

## Requirements

Install the required library with:

```bash
pip install numpy
```

## What I Learned

This project helps explain the basics of probabilistic text classification:

- How text can be converted into word counts
- How class priors influence predictions
- How word likelihoods are calculated per class
- Why Laplace smoothing is important for unseen words
- How Naive Bayes combines many simple probabilities into a final prediction
