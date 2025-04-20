import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# Import project modules (adjust paths as needed)
from config import *
from model import BatchProgramClassifier
from prepare_data_python import get_sequence, get_blocks
from tree_python import ASTNodePython

# ==========
# Dataset
# ==========
# We assume the pickle files (blocks.pkl) contain a DataFrame with at least:
#    - a "code" column holding the block sequences (as produced by Pipeline.generate_block_seqs)
#    - an "actual label" column with labels (human: 1, AI: 2 as in data-converter.py; later we subtract 1 for 0-indexing)

class CodeBlocksDataset(Dataset):
    def __init__(self, pkl_file):
        self.df = pd.read_pickle(pkl_file)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # The model training code subtracts 1 from the label.
        code_blocks = row['code']  # This is a list-of-lists (each block as a list of token indices)
        label = row['actual label'] - 1  # convert label to 0-index (0: AI, 1: human)
        return code_blocks, label

# Collate function that simply groups the samples into a list.
def collate_fn(batch):
    codes, labels = zip(*batch)
    return list(codes), torch.tensor(labels, dtype=torch.long)

# ==========
# File Paths
# ==========
base_dir = os.path.join("AI-Detector", "src", "astnn", "classification", "python", "data")
train_pkl = os.path.join(base_dir, "train", "blocks.pkl")
test_pkl  = os.path.join(base_dir, "test", "blocks.pkl")

train_dataset = CodeBlocksDataset(train_pkl)
test_dataset = CodeBlocksDataset(test_pkl)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print("Train samples:", len(train_dataset))
print("Test samples:", len(test_dataset))

# ==========
# Load Pretrained Embeddings
# ==========
# In your pipeline the word2vec model was saved at:
emb_path = os.path.join(base_dir, "train", "embedding", "node_w2v_" + str(EMBEDDING_SIZE))
from gensim.models.word2vec import Word2Vec
w2v = Word2Vec.load(emb_path).wv
embeddings = np.zeros((w2v.syn0.shape[0] + 1, w2v.syn0.shape[1]), dtype="float32")
embeddings[:w2v.syn0.shape[0]] = w2v.syn0
MAX_TOKENS = w2v.syn0.shape[0]  # This will be used as the "unknown" token index
print("Loaded pretrained embedding with shape:", embeddings.shape)

# ==========
# Instantiate the Model
# ==========
device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")

model = BatchProgramClassifier(
    embedding_dim=EMBEDDING_SIZE,
    hidden_dim=HIDDEN_DIM,
    vocab_size=MAX_TOKENS + 1,
    encode_dim=ENCODE_DIM,
    label_size=LABELS,
    batch_size=BATCH_SIZE,
    use_gpu=USE_GPU,
    pretrained_weight=embeddings
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters())

# ==========
# Training Loop
# ==========
def train_epoch(loader):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch in loader:
        codes, labels = batch
        model.batch_size = len(labels)
        model.hidden = model.init_hidden()
        # Forward pass: the model expects 'codes' to be a list of block sequences.
        outputs = model(codes)
        loss = criterion(outputs, Variable(labels.to(device)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted.cpu() == labels).sum().item()
        total_samples += len(labels)
    return total_loss / total_samples, total_correct / total_samples

def evaluate(loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            codes, labels = batch
            model.batch_size = len(labels)
            model.hidden = model.init_hidden()
            outputs = model(codes)
            loss = criterion(outputs, Variable(labels.to(device)))
            total_loss += loss.item() * len(labels)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted.cpu() == labels).sum().item()
            total_samples += len(labels)
    return total_loss / total_samples, total_correct / total_samples

print("Starting training...")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(train_loader)
    test_loss, test_acc = evaluate(test_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f} | Test Loss={test_loss:.4f}, Test Acc={test_acc:.3f}")

# ==========
# Inference on a Random Python Code Snippet
# ==========
# We use the same pipeline (get_sequence and get_blocks) to preprocess raw code.
# The project’s prepare_data_python.py returns block sequences (lists of token indices).
# For inference, we need to convert raw code to that representation.
def preprocess_for_inference(code_str):
    # Parse the code
    tree = ast.parse(code_str)
    # Get block sequence using the provided function
    blocks = []
    get_blocks(tree, blocks)
    # The blocks list may contain ASTNodePython objects and string markers ("End").
    # Convert each block into a sequence of token indices.
    # We use the pretrained word2vec vocabulary (w2v.vocab) to map tokens.
    def block_to_indices(block):
        if isinstance(block, str):
            # If the marker exists in the vocabulary use its index, otherwise use MAX_TOKENS as unknown.
            return [w2v.vocab[block].index] if block in w2v.vocab else [MAX_TOKENS]
        else:
            token = block.get_token()  # get the token from ASTNodePython
            idx = w2v.vocab[token].index if token in w2v.vocab else MAX_TOKENS
            # Process children recursively
            children_indices = []
            for child in block.children:
                children_indices.extend(block_to_indices(child))
            return [idx] + children_indices
    block_seq = [block_to_indices(b) for b in blocks]
    return block_seq

random_code = """
def random_example(a, b):
    if a > b:
        return a - b
    else:
        return a + b
"""

inference_input = preprocess_for_inference(random_code)
# The model expects a batch (list) of samples.
model.batch_size = 1
model.hidden = model.init_hidden()
model.eval()
with torch.no_grad():
    output = model(inference_input)
    _, pred = torch.max(output, 1)
    # According to your label mapping, 0 corresponds to AI and 1 corresponds to human.
    print("Inference result for random code snippet (1: human, 0: AI):", pred.item())