# --- enable protocol‑5 support in Python 3.6 via pickle5 ---
import pickle5 as pickle
import pandas.io.pickle as pd_pickle

# override the module‑level alias that pandas actually uses
pd_pickle.pkl = pickle
pd_pickle.pickle = pickle
# --- end shim ---

import pandas as pd
import os
from tqdm.auto import tqdm
from config import *
import ast
from prepare_data_python import get_sequence, get_blocks
from gensim.models.word2vec import Word2Vec
tqdm.pandas()


class Pipeline:
    """Pipeline class

    Args:
        ratio ([type]): The ratio for spliting the dataset.
        root (str): Path to the folder containing the data.
    """

    def __init__(self, ratio, root: str):
        self.ratio = ratio
        self.root = root
        self.sources = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.size = None

    # parse source code
    def get_parsed_source(self, input_file: str,
                          output_file: str = None) -> pd.DataFrame:
        """Parse Java code using javalang
        Args:
            input_file (str): Path to the input file
            output_file (str): Path to the output file

        Returns:
            pd.DataFrame: DataFrame with the columns id, code (Java code parsed by
                javalang) and label.
        """

        def parse_program(func):
            try:
                tree = ast.parse(func)
                # tokens = javalang.tokenizer.tokenize(func)
                # parser = javalang.parser.Parser(tokens)
                # tree = parser.parse_member_declaration()
                return tree
            except Exception:
                return None

        input_file_path = os.path.join(self.root, input_file)
        if output_file is None:
            source = pd.read_pickle(input_file_path)
        else:
            source = pd.read_pickle(input_file_path)
            source.columns = ['id', 'code', 'label']

            print(f"Size before parsing: {source.shape}")

            source['code'] = source['code'].progress_apply(parse_program)
            source = source.dropna(subset=['code'])

            print(f"Size after parsing: {source.shape}")

            source.to_pickle(os.path.join(self.root, output_file))
        self.sources = source
        return source

    def split_data(self):
        data = self.sources
        data_num = len(data)

        ratios = [int(r) for r in self.ratio.split(':')]

        train_split = int(ratios[0] / sum(ratios) * data_num)
        val_split = train_split + int(ratios[1] / sum(ratios) * data_num)

        data = data.sample(frac=1, random_state=666)
        train = data.iloc[:train_split]
        dev = data.iloc[train_split:val_split]
        test = data.iloc[val_split:]

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)

        train_path = self.root + 'train/'
        check_or_create(train_path)
        self.train_file_path = train_path + 'train_.pkl'
        train.to_pickle(self.train_file_path)

        dev_path = self.root + 'dev/'
        check_or_create(dev_path)
        self.dev_file_path = dev_path + 'dev_.pkl'
        dev.to_pickle(self.dev_file_path)

        test_path = self.root + 'test/'
        check_or_create(test_path)
        self.test_file_path = test_path + 'test_.pkl'
        test.to_pickle(self.test_file_path)

    def dictionary_and_embedding(self, input_file, size):
        self.size = size
        if not input_file:
            input_file = self.train_file_path
        trees = pd.read_pickle(input_file)
        if not os.path.exists(self.root + 'train/embedding'):
            os.mkdir(self.root + 'train/embedding')

        def trans_to_sequences(ast):
            sequence = []
            get_sequence(ast, sequence)
            return sequence

        corpus = trees['code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['code'] = pd.Series(str_corpus)
        trees.to_csv(self.root + 'train/programs_ns.tsv')

        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=MIN_COUNT, max_final_vocab=VOCAB_SIZE)
        w2v.save(self.root + 'train/embedding/node_w2v_' + str(size))

    def generate_block_seqs(self, data_path, part):
        word2vec = Word2Vec.load(self.root + 'train/embedding/node_w2v_' + str(EMBEDDING_SIZE)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def tree_to_index(node):
            if isinstance(node, str):
                token = node
                result = [vocab[token].index if token in vocab else max_token]
            else:
                token = node.get_token()
                result = [vocab[token].index if token in vocab else max_token]
                children = node.children
                for child in children:
                    result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            get_blocks(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree

        trees = pd.read_pickle(data_path)
        trees['code'] = trees['code'].apply(trans2seq)
        trees.to_pickle(self.root + part + '/blocks.pkl')

    def run(self):
        print('parse source code...')
        # if os.path.exists(os.path.join(self.root, 'ast.pkl')):
        #     self.get_parsed_source(input_file='ast.pkl')
        # else:
        #     self.get_parsed_source(input_file='programs.pkl',
        #                            output_file='ast.pkl')

        self.get_parsed_source(input_file='programs.pkl', output_file = 'ast.pkl')

        print('split data...')
        self.split_data()
        print('train word embedding...')
        self.dictionary_and_embedding(None, EMBEDDING_SIZE)
        print('generate block sequences...')

        self.generate_block_seqs(self.train_file_path, 'train')
        self.generate_block_seqs(self.dev_file_path, 'dev')
        self.generate_block_seqs(self.test_file_path, 'test')


ppl = Pipeline(RATIO, 'data/')
ppl.run()