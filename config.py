# Configs of the other study

VOCAB_SIZE = None # The vocabulary size of Word2Vec, None for no limit.
MIN_COUNT = 3 # Ignores all words with total frequency lower than this.
EMBEDDING_SIZE = 128 # Embedding size of the word vectors.
RATIO = "3:1:1" # The ratio for spliting dataset into training, validation, and testing respectively.
HIDDEN_DIM = 100 # The hidden dimension of the ST-Tree encoder.
ENCODE_DIM = 128 # The hidden dimension of the BiGRU encoder.
LABELS = 104 # The number of the classes for the output.
EPOCHS = 15
BATCH_SIZE = 64
USE_GPU = False


# Configs of our study
GITHUB_TOKEN = "YOUR_GITHUB_PAT_HERE"

REPOS_FILE = "repos.txt"

CLONE_DIR = "cloned_repos"

PREDICTIONS_CSV = "code_predictions.csv"
ISSUES_CSV = "github_issues.csv"
LINKS_CSV = "code_issue_links.csv"
SAMPLED_LINKS_CSV = "sampled_links.csv"

# Index directories
CODE_INDEX_DIR = "lucene_code_index"
ISSUE_INDEX_DIR = "lucene_issue_index"

START_YEAR = 2020
END_YEAR = 2025

# Sampling parameters
SAMPLE_SIZE = 100
SAMPLE_MOE = 0.10

import os
def get_github_token():
    token = os.environ.get("GITHUB_TOKEN", GITHUB_TOKEN)
    if not token or token == "YOUR_GITHUB_PAT_HERE":
        raise ValueError("GitHub token not configured. Set GITHUB_TOKEN environment variable or edit config.py")
    return token