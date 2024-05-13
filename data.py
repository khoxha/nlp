from typing import Dict, List, Tuple

import torch
from more_itertools import windowed


def read_data_from_file(path) -> List[Tuple[List[str], str]]:
    """
    Read data from a file and return a list of (sentence, label) tuples.
    You know this function from exercise 1.
    :param path: Path to the file
    """
    data = []
    with open(path) as file:
        for line in file.readlines():
            line = line.strip()
            label = line.split(" ")[0][9:]
            text = line.split(" ")[1:]
            data.append((text, label))

    return data


def make_label_dictionary(data) -> Dict[str, int]:
    """
    Make a dictionary of labels.
    :param data: List of (sentence, label) tuples
    :return: A dictionary of string keys and index values
    """
    label_to_ix = {}
    for _, label in data:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)
    return label_to_ix




def make_label_vector(label, label_to_ix) -> torch.Tensor:
    """
    Make a label vector from a label.
    :param label: A label string
    :param label_to_ix: A dictionary mapping labels to indices
    :return: A PyTorch tensor
    """
    # Convert label to index
    label_idx = label_to_ix[label]
    # Create a tensor with the index
    label_tensor = torch.tensor(label_idx).unsqueeze(0).long()
    #print(f"Label tensor shape: {label_tensor.shape}")
    return label_tensor







def make_bow_dictionary(data) -> Dict[str, int]:
    """
    Make a dictionary of words. This function is used in the Bag-of-Words model.
    :param data: List of (sentence, label) tuples
    :return: A dictionary of string keys and index values
    """
    word_to_ix = {}
    for sent, _ in data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    word_to_ix["<UNK>"] = len(word_to_ix)
    return word_to_ix


def make_bow_vector(sentence, word_to_ix) -> torch.Tensor:
    """
    Make a Bag-of-Words vector from a sentence.
    :param sentence: A list of words
    :param word_to_ix: A dictionary mapping words to indices
    :return: A PyTorch tensor
    """
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        if word in word_to_ix:
            vec[word_to_ix[word]] += 1
        else:
            vec[word_to_ix["<UNK>"]] += 1
    return vec.view(1, -1)



def make_ngram_dictionary(data, unk_threshold: int = 0, max_ngrams: int = 1) -> Dict[str, int]:
    """
    Makes a dictionary of words given a list of tokenized sentences.
    :param data: List of (sentence, label) tuples
    :param unk_threshold: All words below this count threshold are excluded from dictionary and replaced with UNK
    :return: A dictionary of string keys and index values
    """
    ngram_frequencies = {}
    for sent, _ in data:
        # Go over all n-gram sizes (including 1)
        for ngram_size in range(1, max_ngrams + 1):
            # Move a window over the text
            for ngram in windowed(sent, ngram_size):
                # Skip NoneType ngrams
                if None in ngram:
                    continue
                ngram_str = ' '.join(ngram)
                if ngram_str not in ngram_frequencies:
                    ngram_frequencies[ngram_str] = 1
                else:
                    ngram_frequencies[ngram_str] += 1

    # Filter out n-grams below the threshold
    filtered_ngrams = {ngram: freq for ngram, freq in ngram_frequencies.items() if freq >= unk_threshold}

    # Assign indices to each distinct n-gram
    ngram_to_ix = {ngram: index for index, (ngram, _) in enumerate(filtered_ngrams.items())}

    return ngram_to_ix


def make_ngram_vectors(sentence, word_to_ix, max_ngrams: int = 1) -> torch.Tensor:
    """
    Make a list of one-hot vectors from a sentence.
    We use n-grams here.
    :param sentence: A list of words
    :param word_to_ix: A dictionary mapping words to indices
    :param max_ngrams: The maximum n-gram size
    :return: A PyTorch tensor
    """
    # Initialize a list to store the one-hot vectors
    ngram_vectors = []

    # Iterate over all n-gram sizes (up to max_ngrams)
    for ngram_size in range(1, max_ngrams + 1):
        # Iterate over all possible n-grams in the sentence
        for start in range(len(sentence) - ngram_size + 1):
            ngram = ' '.join(sentence[start:start + ngram_size])
            # Check if the n-gram exists in the word_to_ix dictionary
            if ngram in word_to_ix:
                # If yes, create a one-hot vector with the corresponding index
                one_hot = torch.zeros(len(word_to_ix))
                one_hot[word_to_ix[ngram]] = 1
                ngram_vectors.append(one_hot)

    # Check if ngram_vectors is empty before stacking
    if not ngram_vectors:
        # If empty, return a tensor of zeros
        return torch.zeros(1, len(word_to_ix))
    else:
        # If not empty, stack the tensors and convert to long type
        return torch.stack(ngram_vectors).long()




def make_word_dictionary_from_pretrained(embedding_model) -> Dict[str, int]:
    """
    Make a dictionary of words from a pretrained embedding model.
    :param embedding_model: A pretrained word embedding model
    :return: A dictionary of string keys and index values
    """
    word_dictionary = {}
    for index, word in enumerate(embedding_model.index_to_key):
        word_dictionary[word] = index
    return word_dictionary


def load_odd_one_out():
    """
    Load the odd one out dataset.
    """
    data = []
    with open("data/odd_one_out.txt") as file:
        for line in file.readlines():
            words = line.strip().split(" ")
            words = [word.lower() for word in words]
            data.append(words)

    return data
