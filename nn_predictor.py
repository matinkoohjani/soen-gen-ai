import ast
import torch
import torch.nn.functional as F
import numpy as np
from gensim.models.word2vec import Word2Vec
from model import BatchProgramClassifier
from config import *
from prepare_data_python import get_blocks
from tree_python import ASTNodePython


class ProgramClassifierWrapper:
    def __init__(self,
                 w2v_path: str = f"AI-Detector-Dependency/src/astnn/classification/python/data/train/embedding/node_w2v_{EMBEDDING_SIZE}",
                 state_dict_path: str = "AI-Detector-Dependency/src/astnn/classification/python/best_model_state_dict.pth",
                 use_gpu: bool = USE_GPU):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        self.word2vec = Word2Vec.load(w2v_path).wv
        self.embeddings = np.zeros(
            (self.word2vec.vectors.shape[0] + 1, self.word2vec.vectors.shape[1]),
            dtype="float32"
        )
        self.embeddings[: self.word2vec.vectors.shape[0]] = self.word2vec.vectors

        self.MAX_TOKENS = self.word2vec.vectors.shape[0]
        self.EMBEDDING_DIM = self.word2vec.vectors.shape[1]
        self.vocab = self.word2vec.key_to_index

        self.model = BatchProgramClassifier(
            self.EMBEDDING_DIM,
            HIDDEN_DIM,
            self.MAX_TOKENS + 1,
            ENCODE_DIM,
            LABELS,
            batch_size=1,
            use_gpu=use_gpu,
            pretrained_weight=self.embeddings
        )
        self.model.load_state_dict(
            torch.load(state_dict_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    def tree_to_index(self, node):

        if isinstance(node, str):
            return [self.vocab.get(node, self.MAX_TOKENS)]
        if not isinstance(node, ASTNodePython):
            raise ValueError(f"Unsupported node type: {type(node)}")
        token = node.get_token()
        result = [self.vocab.get(token, self.MAX_TOKENS)]
        for child in node.children:
            result.append(self.tree_to_index(child))
        return result

    def prepare(self, code: str):

        ast_tree = ast.parse(code)
        blocks = []
        get_blocks(ast_tree, blocks)
        index_tree = [self.tree_to_index(b) for b in blocks]
        # Batch of size 1
        return [index_tree]

    def predict(self, code: str):
        batch_data = self.prepare(code)
        self.model.batch_size = 1
        self.model.hidden = self.model.init_hidden()

        with torch.no_grad():
            output = self.model(batch_data)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            pred = output.argmax(dim=1).item()
        return pred, probs

    def predict_file(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        return self.predict(code)
