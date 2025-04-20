import sys
import ast
import torch
import numpy as np
from gensim.models.word2vec import Word2Vec
from model import BatchProgramClassifier
from config import *
from prepare_data_python import get_blocks
from tree_python import ASTNodePython

# ------------------------------------------------------------------------------
# 1. Parse command-line argument for input file
# ------------------------------------------------------------------------------

input_file = "/Users/matin/Desktop/research/pydriller/projects/gitpython_demo.py"
try:
    with open(input_file, "r", encoding="utf-8") as f:
        python_code = f.read()
except IOError as e:
    print(f"Error reading file {input_file}: {e}")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 2. Load the pre-trained Word2Vec embeddings (vocabulary)
# ------------------------------------------------------------------------------
w2v_path = "data/train/embedding/node_w2v_" + str(EMBEDDING_SIZE)
word2vec = Word2Vec.load(w2v_path).wv

# Build the numpy embedding matrix (same as during training)
embeddings = np.zeros((word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32")
embeddings[: word2vec.vectors.shape[0]] = word2vec.vectors
MAX_TOKENS = word2vec.vectors.shape[0]
EMBEDDING_DIM = word2vec.vectors.shape[1]

# Use the new Gensim API for vocabulary mapping:
vocab = word2vec.key_to_index

# ------------------------------------------------------------------------------
# 3. Reconstruct the model and load the saved state dictionary
# ------------------------------------------------------------------------------
model = BatchProgramClassifier(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE, USE_GPU, embeddings)
device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
model.to(device)

state_dict_path = "best_model_state_dict.pth"
model.load_state_dict(torch.load(state_dict_path, map_location=device))
model.eval()  # Set the model to evaluation mode

# ------------------------------------------------------------------------------
# 4. Prepare the input: Parse the code, extract blocks and convert tokens to indices
# ------------------------------------------------------------------------------
try:
    ast_tree = ast.parse(python_code)
except Exception as e:
    print(f"Error parsing the Python code: {e}")
    sys.exit(1)

blocks = []
get_blocks(ast_tree, blocks)

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

index_tree = []
for b in blocks:
    index_tree.append(tree_to_index(b, vocab, MAX_TOKENS))

# Create a batch (list) of one sample:
batch_data = [index_tree]

# ------------------------------------------------------------------------------
# 5. Run prediction with the loaded model
# ------------------------------------------------------------------------------
model.batch_size = 1  # We have a single sample
model.hidden = model.init_hidden()

with torch.no_grad():
    output = model(batch_data)
    max_scores, predicted = torch.max(output.data, 1)
    prediction = predicted.item()

print("Max Scores:", max_scores)
print("Predicted Classes:", prediction)

import torch.nn.functional as F

# Disable gradient calculation during inference
with torch.no_grad():
    # Pass the prepared batch_data through the model
    output = model(batch_data)
    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(output, dim=1)
    print("Probabilities:", probabilities)

    # Optionally, you can extract the predicted class:
    _, predicted = torch.max(output.data, 1)
    print("Predicted class:", predicted.item())
