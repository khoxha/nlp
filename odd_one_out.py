from itertools import combinations
from scipy.spatial import distance
from gensim.models.keyedvectors import Word2VecKeyedVectors
from data import load_odd_one_out

def main():
    # Load the pretrained embeddings
    pretrained_word_embeddings_file = "pretrained_embeddings/glove-wiki-gigaword-50.txt"
    data = load_odd_one_out()
    pretrained_embeddings = Word2VecKeyedVectors.load_word2vec_format(
        pretrained_word_embeddings_file
    )

    # Distance functions to try
    distance_functions = {
        "Cosine": distance.cosine,
        "Euclidean": distance.euclidean,
        "Manhattan": distance.cityblock
    }

    # Iterate over the examples
    for example in data:
        print("Example:", example)
        print("Calculating distances using different distance functions:")

        # Calculate distances for each distance function
        for distance_function_name, distance_function in distance_functions.items():
            # Calculate pairwise distances between all words in the example
            distances = {}
            for word1, word2 in combinations(example, 2):
                if word1 in pretrained_embeddings and word2 in pretrained_embeddings:
                    distances[(word1, word2)] = distance_function(
                        pretrained_embeddings[word1], pretrained_embeddings[word2]
                    )

            # Check if distances dictionary is empty
            if distances:
                print(f"Distance function: {distance_function_name}")
                for word1, word2 in combinations(example, 2):
                    if (word1, word2) in distances:
                        print(f"Distance between {word1} and {word2}: {distances[(word1, word2)]}")
                print("----------")

                # Find the odd one out
                word1, word2 = min(distances, key=distances.get)
                odd_one_out = next(word for word in example if word != word1 and word != word2)
                print(f"Odd one out in {example} is {odd_one_out}.")
                print(10 * "-")
            else:
                print(f"No word pairs found for distance calculation with {distance_function_name}.")
                print("----------")


if __name__ == "__main__":
    main()
