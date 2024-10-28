import pandas as pd
import numpy as np

def train_naive_bayes(data: pd.DataFrame):
  # Get the number of docs and classes of the data
  n = len(data)
  classes = data['label'].unique().tolist()

  # Construct vocabuary, and count of each word and total word count in each class
  vocab = set()
  vocab_counts = {}
  class_word_counts = {}
  # Loop over each sentence-class pair
  for sentence, c in zip(data['sentence'].tolist(), data['label'].tolist()):
    for word in sentence.split(): # Loop over each word
      vocab_counts[(word,c)] = vocab_counts.get((word,c),0) + 1 # Increment count of word in that class
      class_word_counts[c] = class_word_counts.get(c,0) + 1 # Increment number of words in that class
      vocab.add(word) # Add word to the vocabulary set

  # Calculate log versions of prior probabilities p(c) and posterior probabiltiies for each word p(w|c) 
  log_prior = {}
  log_likelihood = {}
  for c in classes:
    log_prior[c] = np.log(np.sum(data['label'] == c) / n)
    for word in vocab:
      log_likelihood[(word,c)] = np.log((vocab_counts.get((word,c),0) + 1) / (class_word_counts[c] + len(vocab)))

  return log_prior, log_likelihood, vocab


def test_naive_bayes(test_doc: str, log_prior: dict, log_likelihood: dict, classes: list[int], vocab: set):
  # Initialize all scores to 0
  score = [0]*len(classes)
  # Loop over each class and calculate the score of this class given the test doc
  for c_i,c in enumerate(classes):
    score[c_i] = log_prior[c] # Initialize score to log prior probability
    # Add likelihood of each word in the sentence given current class (if it was in the vocabulary)
    for word in test_doc.lower().split():
      if word in vocab:
        score[c_i] += log_likelihood[(word,c)]
  # Return class with the max score
  return classes[np.argmax(score)]
  
