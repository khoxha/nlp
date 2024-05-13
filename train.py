import os
import json
import time

from tqdm import tqdm
from typing import List, Tuple, Dict
from itertools import product
from statistics import mean

import torch
import torch.optim as optim
from gensim.models.keyedvectors import Word2VecKeyedVectors

from data import (
    read_data_from_file,
    make_label_dictionary,
    make_label_vector,
    make_ngram_dictionary,
    make_ngram_vectors,
    make_word_dictionary_from_pretrained,
)
from models import WordEmbeddingClassifier
from plot import plot_runs


def train(
    model: torch.nn.Module,
    vectorized_training_data: Tuple[torch.Tensor, torch.Tensor],
    num_epochs: int,
    learning_rate: float,
):
    # Define the loss and a simple SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Create an iterator over the training data
    loss_log: List[float] = []
    loss_curve: List[float] = []
    metric_curve: List[float] = []
    epoch_accuracy: float = 0.0
    epoch_loss: float = 0.0

    # Create an iterator over the training data
    data_iterator = iter(vectorized_training_data * num_epochs)
    training_steps = len(vectorized_training_data) * num_epochs
    pbar = tqdm(range(1, training_steps + 1))
    # Go over the training dataset
    for training_step in pbar:

        # Get next instance and label
        instance, target = next(data_iterator)
        # Remember that PyTorch accumulates gradients
        # We need to clear them out before each instance
        model.zero_grad()

        # Run our forward pass
        outputs: Dict = model(instance, target)
        loss = outputs["loss"]

        # Compute the loss, gradients, and update the parameters by calling optimizer.step()
        loss_log.append(loss.item())
        loss.backward()
        optimizer.step()

        # Log mean loss every 1000 steps
        if training_step % 1000 == 0:
            loss_curve.append((training_step, mean(loss_log)))
            loss_log = []

        # Evaluate the model every epoch
        if training_step % len(vectorized_training_data) == 0:
            eval_output = model.evaluate(vectorized_training_data)
            epoch_accuracy = eval_output["accuracy"]
            epoch_loss = eval_output["validation_loss"].item()
            metric_curve.append((training_step, epoch_accuracy))

        # Update progress bar
        description = f"epoch: {training_step // len(vectorized_training_data)} | step: {training_step} | epoch loss: {epoch_loss:.2f} | epoch accuracy: {epoch_accuracy:.2f}"
        pbar.set_description(description)

    return {"loss_curve": loss_curve, "metric_curve": metric_curve}


def main(
    learning_rate: float,
    max_ngrams: int,
    hidden_size: int,
    num_epochs: int,
    unk_threshold: int,
    pretrained_word_embeddings_file: str = None,
):
    # Set seed for reproducibility
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training and test datasets
    training_data = read_data_from_file("data/training_data.txt")
    test_data = read_data_from_file("data/validation_data.txt")

    # Make word dictionary either from pretrained embeddings or from scratch
    if pretrained_word_embeddings_file:
        pretrained_embeddings = Word2VecKeyedVectors.load_word2vec_format(
            pretrained_word_embeddings_file
        )
        hidden_size = pretrained_embeddings.vector_size
        word_dictionary = make_word_dictionary_from_pretrained(pretrained_embeddings)
        max_ngrams = 1  # We only use unigrams for pretrained embeddings
    else:
        word_dictionary = make_ngram_dictionary(
            training_data, max_ngrams=max_ngrams, unk_threshold=unk_threshold
        )

    # Make label dictionary
    label_dictionary = make_label_dictionary(training_data)

    # Vectorize training and test data
    vectorized_training_data: List[torch.Tensor] = [
        (
            make_ngram_vectors(instance, word_dictionary, max_ngrams=max_ngrams).to(
                device
            ),
            make_label_vector(label, label_dictionary).to(device),
        )
        for instance, label in training_data
    ]

    vectorized_test_data: List[torch.Tensor] = [
        (
            make_ngram_vectors(instance, word_dictionary, max_ngrams=max_ngrams).to(
                device
            ),
            make_label_vector(label, label_dictionary).to(device),
        )
        for instance, label in test_data
    ]

    # Initialize model, loss function and optimizer
    model = WordEmbeddingClassifier(
        vocab_size=len(word_dictionary),
        hidden_size=hidden_size,
        num_labels=len(label_dictionary),
        pretrained_embeddings=(
            pretrained_embeddings if pretrained_word_embeddings_file else None
        ),
    )

    # Move model to GPU if available
    model.to(device)

    # Train the model
    start_time = time.time()
    train_outputs = train(
        model,
        vectorized_training_data,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )
    end_time = time.time()

    # Evaluate model
    eval_output = model.evaluate(
        vectorized_test_data,
    )

    return {
        "learning_rate": learning_rate,
        "max_ngrams": max_ngrams,
        "hidden_size": hidden_size,
        "unk_threshold": unk_threshold,
        "num_epochs": num_epochs,
        "pretrained_word_embeddings_file": pretrained_word_embeddings_file,
        "test_accuracy": eval_output["accuracy"],
        "test_loss": eval_output["validation_loss"].item(),
        "loss_curve": train_outputs["loss_curve"],
        "metric_curve": train_outputs["metric_curve"],
        "runtime (in seconds)": end_time - start_time,
    }


if __name__ == "__main__":
    # All hyperparameters
    # TODO: After you implemnted task 1.1 and 1.2, try out different hyperparameters!
    learning_rates = [0.1]
    max_ngrams = [2]
    hidden_sizes = [100]
    num_epochs = [2]
    unk_thresholds = [0]
    pretrained_word_embeddings_files = [None]

    results = []

    # product makes the cartesian product of all hyperparameters
    for config in product(
        learning_rates,
        max_ngrams,
        hidden_sizes,
        num_epochs,
        unk_thresholds,
        pretrained_word_embeddings_files,
    ):
        results.append(main(*config))

    if not os.path.exists("results"):
        os.makedirs("results")

    for result in results:
        i = 0
        while os.path.exists(f"results/run_%s.json"):
            i += 1
        filename = f"results/run_{i}.json"
        with open(filename, "w") as f:
            json.dump(result, f)

    plot_runs()
