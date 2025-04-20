import sys
import os
import ast
import torch
import numpy as np
import torch.nn.functional as F
from gensim.models.word2vec import Word2Vec
from model import BatchProgramClassifier
from config import *
from prepare_data_python import get_blocks
from tree_python import ASTNodePython


def tree_to_index(node, vocab, max_token):
    """
    Convert a block (AST node or string) into its index representation using the vocabulary.
    If the token is not present in vocab, use max_token.
    """
    if isinstance(node, str):
        token = node
        return [vocab.get(token, max_token)]
    else:
        token = node.get_token()
        result = [vocab.get(token, max_token)]
        for child in node.children:
            result.append(tree_to_index(child, vocab, max_token))
        return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python test.py <directory>")
        sys.exit(1)
    input_dir = sys.argv[1]

    # ------------------------------------------------------------------------------
    # 1. Load the pre-trained Word2Vec embeddings and set up vocabulary
    # ------------------------------------------------------------------------------
    w2v_path = "data/train/embedding/node_w2v_" + str(EMBEDDING_SIZE)
    word2vec = Word2Vec.load(w2v_path).wv
    embeddings = np.zeros((word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32")
    embeddings[: word2vec.vectors.shape[0]] = word2vec.vectors
    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]
    # Using the new Gensim API vocabulary mapping:
    vocab = word2vec.key_to_index

    # ------------------------------------------------------------------------------
    # 2. Reconstruct the model and load the saved state dictionary
    # ------------------------------------------------------------------------------
    model = BatchProgramClassifier(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE, USE_GPU,
                                   embeddings)
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    model.to(device)
    state_dict_path = "best_model_state_dict.pth"
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.eval()

    # ------------------------------------------------------------------------------
    # 3. Process all .py files in the provided directory recursively
    # ------------------------------------------------------------------------------
    prediction_counts = {i: 0 for i in range(LABELS)}

    for root_dir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root_dir, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        python_code = f.read()
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    continue

                try:
                    ast_tree = ast.parse(python_code)
                except Exception as e:
                    print(f"Error parsing file {file_path}: {e}")
                    continue

                try:
                    blocks = []
                    get_blocks(ast_tree, blocks)
                    index_tree = [tree_to_index(b, vocab, MAX_TOKENS) for b in blocks]
                    # Create a batch (list) of one sample:
                    batch_data = [index_tree]
                    model.batch_size = 1
                    model.hidden = model.init_hidden()

                    with torch.no_grad():
                        output = model(batch_data)
                        # Retrieve the predicted class index
                        _, predicted = torch.max(output.data, 1)
                        prediction = predicted.item()
                    prediction_counts[prediction] += 1
                    print(f"File: {file_path} -> Predicted class: {prediction}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue

    # ------------------------------------------------------------------------------
    # 4. Print the final prediction report
    # ------------------------------------------------------------------------------
    print("\nPrediction Report:")
    for cls, count in prediction_counts.items():
        print(f"Class {cls}: {count} instance(s)")


if __name__ == "__main__":
    main()