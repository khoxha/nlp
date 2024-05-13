# Exercise 2

Clone this repository via Github Classrooms: https://classroom.github.com/a/Re_pkagu

Navigate to the cloned folder:
```bash
cd nlp-exercise-2-solution
```

Activate your environment (refer to exercise 1 for setting up your environment):
```bash
conda activate nlpcourse
```

# Assignments

You will need to complete functions and class in this exercise. Check the exercise sheet for detailed instructions. We marked all places with the `TODO` label where you need to implement something. Places without the `TODO` labels do not need to be adjusted.
You need to complete the tasks such it is possible to run each task like:
```bash
python train.py
```

The repository is structured in the following way:
```bash
|-data/ # the data folder
|-pretrained_embeddings/
|-results/ # where the results will be saved
|-data.py # Contains all data-related functions (i.e. turn words into vectors)
|-models.py # Contains all models (WordEmbeddingClassifier and BoWClassifier from previous exercise)
|-odd_one_out.py # This script is for the 2nd task in which you do odd-one-out
|-plot.py # This script plots loss curves and validation metrics
|-train.py # Run this script to train your FastText classifier
```

## Task 1

In this task, you need to implement a FastText classifier. To complete this exercise, you need to:

1. Complete the `WordEmbeddingClassifier` class.
2. Implement the `make_ngram_dictionary()` and `make_ngram_vectors()` to obtain the word dictionary.
3. Try out 10 different hyperparameter settings and analyze the results in the `results` directory.

## Task 2
Given lists of words, find the word that does not belong in the list.
1. Load a pre-trained embedding (look at `pretrained_embeddings` for GloVe embeddings or download [pretrained FastText embeddings](https://drive.google.com/file/d/1Bqs0cYXHTNuhwBE_EToKlVp1JbH7H1ZP/view)) and embedded each word of a list. Use some distance metric to find out the word that does not belong into the list.
