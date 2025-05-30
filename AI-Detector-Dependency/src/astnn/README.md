# ASTNN--A Novel Neural Source Code Representation based on Abstract Syntax Tree
This repository includes the code and experimental data in our paper entitled "A Novel Neural Source Code Representation based on Abstract Syntax Tree" published in ICSE'2019. It can be used to encode code fragments into supervised vectors for various source code related tasks. We have applied our neural source code representation to two common tasks: source code classification and code clone detection. It is also expected to be helpful in more tasks.

### Requirements
+ python 3.6.7<br> Note: the version should be exactly the same to properly load pickle files.
+ pandas 0.20.3<br>
+ gensim 3.5.0<br>
+ scikit-learn 0.19.1<br>
+ pytorch 1.0.0<br> (The version used in our paper is 0.3.1 and source code can be cloned by specifying the v1.0.0 tag if needed)
+ pycparser 2.18<br>
+ javalang 0.11.0<br>
+ RAM 16GB or more
+ GPU with CUDA support is also needed
+ BATCH_SIZE should be configured based on the GPU memory size

### How to install
Install all the dependent packages via pip:

	$ pip install pandas==0.20.3 gensim==3.5.0 scikit-learn==0.19.1 pycparser==2.18 javalang==0.11.0
 
Install pytorch according to your environment, see https://pytorch.org/ 


### Source Code Classification
1. `cd astnn`
2. run `python pipeline.py` to generate preprocessed data.
3. run `python train.py` for training and evaluation

### Code Clone Detection

 1. `cd clone`
 2. run `python pipeline.py --lang c` or `python pipeline.py --lang java` to generate preprocessed data for the two datasets.
 2. run `python train.py --lang c` to train on OJClone, `python train.py --lang java` on BigCLoneBench respectively.

### How to use it on your own dataset

Please refer to the `pkl` files in the corresponding directories of the two tasks. These files can be loaded by `pandas`.
For example, to realize clone detection, you need to replace the two files in /clone/data/java, bcb_pair_ids.pkl and bcb_funcs_all.tsv.
Specifically, the data format of bcb_pair_ids.pkl  is "id1, id2, label", where id1/2 correspond to the id in  bcb_funcs_all.tsv and label indicates whether they are clone or which clone type (i.e., 0 and 1-5, or 0 and 1 in a non-type case).
The data format of bcb_funcs_all.tsv is "id, function".

### Expanded Applications
* Unsupervised Vector Representation. Utilizing Word2Vec within our framework allows for the straightforward generation of vector representations through the `gru_out` tensor (shape of `B*H`) located in model.py. This approach facilitates the embedding of words into a dense vector space, where semantic similarities translate into spatial closeness. However, it's important to note that the effectiveness of these vector representations can vary significantly across different tasks. 
* Enhancing Sequence-to-Sequence Models. The ASTNN model can serve as an encoder in sequence-to-sequence tasks, such as code summarization. By extracting the hidden states from the `gru_out` tensor before pooling (shape of `B*L*H`) in model.py, ASTNN captures the syntactical and sequential information within the code. These captured sequences are then ready to be fed into a decoder.
 
### Citation
  If you find this code useful in your research, please, consider citing our paper:
  > @inproceedings{zhang2019novel,  
  title={A novel neural source code representation based on abstract syntax tree},  
  author={Zhang, Jian and Wang, Xu and Zhang, Hongyu and Sun, Hailong and Wang, Kaixuan and Liu, Xudong},    
  booktitle={Proceedings of the 41st International Conference on Software Engineering},  
  pages={783--794},  
  year={2019},  
  organization={IEEE Press}  
}
