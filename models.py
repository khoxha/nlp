from abc import abstractmethod
from typing import Optional, List
from tqdm import tqdm

import torch
import torch.nn.functional as F


from gensim.models.keyedvectors import Word2VecKeyedVectors


class Classifier(torch.nn.Module):
    """Abstract class for a classifier. You need to implement the forward method.
    Evaluation method is shared of all classifiers."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        loss_function: torch.nn.Module = torch.nn.NLLLoss,
    ):
        super(Classifier, self).__init__()
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.loss_function = loss_function()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def evaluate(
        self,
        test_data: List,
    ) -> float:
        tp: int = 0
        fp: int = 0

        validation_loss: float = 0.0

        # Create an iterator over the training data
        data_iterator = iter(test_data)
        evaluation_steps = len(test_data)
        pbar = tqdm(range(1, evaluation_steps + 1))

        with torch.no_grad():
            for evaluation_step in pbar:

                # Get next instance and label
                vector, label = next(data_iterator)

                # Run our forward pass
                outputs = self.forward(vector, label)

                # Extract the log probabilities
                log_probs = outputs["log_probs"]

                # Check if the predicted label is correct
                if torch.argmax(log_probs).item() == label.item():
                    tp += 1
                else:
                    fp += 1

                # Extract the loss
                validation_loss += outputs["loss"]

                # Update the progress bar
                description = f"Evaluation | {evaluation_step}/{len(test_data)}"
                pbar.set_description(description)

            # Calculate the accuracy
            accuracy = tp / (tp + fp)
            validation_loss /= len(test_data)

            return {"accuracy": accuracy, "validation_loss": validation_loss}


class WordEmbeddingClassifier(Classifier):
    """This is a simple FastText / WordEmbedding classifier."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        hidden_size: int,
        pretrained_embeddings: Optional[Word2VecKeyedVectors] = None,
    ):
        super().__init__(vocab_size=vocab_size, num_labels=num_labels)
        if pretrained_embeddings is not None:
            # If pretrained embeddings are provided, initialize the embedding layer with them
            embedding_matrix = pretrained_embeddings.vectors
            self.embedding = torch.nn.Embedding.from_pretrained(embedding_matrix)
        else:
            # If no pretrained embeddings are provided, initialize the embedding layer randomly
            self.embedding = torch.nn.Embedding(vocab_size, hidden_size)

        self.linear = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, one_hot_sentence, target):
        embedded = self.embedding(one_hot_sentence.long())  # Convert to torch.long
        pooled = torch.mean(embedded, dim=0)
        logits = self.linear(pooled)
        log_probs = F.log_softmax(logits, dim=0)
        
        # One-hot encode the target tensor
        target_one_hot = F.one_hot(target, num_classes=self.num_labels)
        
        # Calculate the loss
        loss = self.loss_function(log_probs.unsqueeze(0), target_one_hot)

        return {"loss": loss, "log_probs": log_probs}




class BoWClassifier(Classifier):
    """This is a simple Bag-of-Words classifier. You know this class from the previous exercise."""

    def __init__(self, vocab_size: int, num_labels: int):
        super().__init__(vocab_size=vocab_size, num_labels=num_labels)
        self.linear = torch.nn.Linear(self.vocab_size, self.num_labels)

    def forward(self, bow_vec, target):
        features = self.linear(bow_vec)
        log_probs = F.log_softmax(features, dim=1)
        loss = self.loss_function(log_probs, target)
        return {"loss": loss, "log_probs": log_probs}
