# Kernel-Based Structural-Temporal Cascade Learningfor Popularity Prediction
This repo provides a reference implementation of **CasKernel**

[comment]: <> ([comment]: <> &#40;>  Quantifying the Scientific Impact via Heterogeneous Dynamical Graph Neural Network  &#41;)

[comment]: <> ([comment]: <> &#40;>  [Xovee Xu]&#40;https://xovee.cn&#41;, Fan Zhou, Ce Li, Goce Trajcevski, Ting Zhong, and Kunpeng Zhang &#41;)

[comment]: <> ([comment]: <> &#40;>  Submitted for review  &#41;)

## Requirements

The code was tested with `Python 3.7`, `tensorflow-gpu 2.4.0`, `torch 1.0.1` and `Cuda 11.0.221`. Install the dependencies via Anaconda: 

[comment]: <> (```shell)

[comment]: <> (# create conda virtual environment)

[comment]: <> (conda create --name SIHDGNN python=3.7 cudatoolkit=11.0.221 cudnn=8.0.4 pytorch=1.0.1 torchvision=0.2.2 -c pytorch)

[comment]: <> (# activate environment)

[comment]: <> (conda activate SIHDGNN)

[comment]: <> (# install other dependencies)

[comment]: <> (pip install -r requirements.txt)

[comment]: <> (```)

[comment]: <> (## Datasets)

[comment]: <> (Dataset can be downloaded in [Google Drive]&#40;https://drive.google.com/drive/folders/1JPXdSi23VS1lt0O_clxzNvaHgRl9iaIY?usp=sharing&#41;.)

[comment]: <> (You can access the original APS dataset [here]&#40;https://journals.aps.org/datasets&#41;. &#40;Released by *American Physical Society*, obtained at Jan 17, 2019&#41;)



[comment]: <> (# Run the code)

[comment]: <> (For a given scientific dataset, you should:)

[comment]: <> (1. Construct a heterogeneous graph)

[comment]: <> (2. Get node embeddings)

[comment]: <> (3. Generate scientific information cascades)

[comment]: <> (4. Training & evaluating)

[comment]: <> (## Construct heterogeneous graph)

[comment]: <> (This stage may costs a large amount of RAM &#40;~64GB with millions of nodes/edges in graph&#41;, delete some nodes/edges to save space.)

[comment]: <> (### Run scripts:)

[comment]: <> (```shell)

[comment]: <> (# build a heterogeneous graph)

[comment]: <> (python graph_sample.py)

[comment]: <> (# heterogeneous neighboring node sampling)

[comment]: <> (python rwr.py)

[comment]: <> (```)

[comment]: <> (## Generate node embeddings)

[comment]: <> (After graph construction, we now learn node embeddings via a heterogeneous graph neural network. )

[comment]: <> (### Input fiels:)

[comment]: <> (1. `a_p_list_train.txt`: `author:paper1,paper2,...`, author and papers written by this author)

[comment]: <> (2. `p_a_list_train.txt`: `original_paper:author1,author2,author3,...`, paper and its authors)

[comment]: <> (3. `p_p_citation_list.txt`: `original_paper:paper1,paper2,...`, paper and its citation papers)

[comment]: <> (4. `v_p_list_train.txt`: `venue:paper1,paper2,...`, venue and papers published on this venue)

[comment]: <> (5. `node_net_embedding`: each line is an embedding of a node, trained by DeepWalk)

[comment]: <> (6. `het_neigh_train.txt` and `het_random_walk.txt`: sample neighbors through random walk)

[comment]: <> (```shell script)

[comment]: <> (> cd ./codes/gnn)

[comment]: <> (> python gene_node_embeddings.py)

[comment]: <> (```)

[comment]: <> (## Generate scientific information cascades)

[comment]: <> (Once we got the node embeddings, we can generate cascades and corresponding training/validation/test data.)

[comment]: <> (### Input files for paper prediction:)

[comment]: <> (Here we only include the files related to the paper prediction, followed by the author prediction)

[comment]: <> (1. `node_embedding.txt`: each line is an embedding of a node)

[comment]: <> (2. `p2_cited_citing_lst.txt`: `original_paper:citing_paper1,citing_paper2,...`  &#40;including 2 years of citing papers&#41;)

[comment]: <> (3. `p20_cited_citing_lst.txt`: `original_paper:num_citations`  &#40;including 20 years of citations&#41;)

[comment]: <> (4. `p_a_lst_train.txt`: `original_paper:author1,author2,author3,...`, paper and its authors)

[comment]: <> (5. `p_v.txt`: `original_paper,venue`)

[comment]: <> (### Input files for author prediction:)

[comment]: <> (1. `node_embedding.txt`: each line is an embedding of a node)

[comment]: <> (2. `a2_cited_citing_lst.txt`: `original_author:citing_paper1,citing_paper2,...`  &#40;including 2 years of citing papers&#41;)

[comment]: <> (3. `a20_cited_citing_lst.txt`: `original_author:num_citations`  &#40;including 20 years of citations&#41;)

[comment]: <> (4. `p_a_lst_train.txt`: `original_paper:author1,author2,author3,...`, paper and its authors)

[comment]: <> (5. `paper_addition.pkl`: `original_author:[&#40;publication,[citation1,citation2...]&#41;]`, paper and its publication and citation)

[comment]: <> (### Paper Prediction Run scripts:)

[comment]: <> (```shell script)

[comment]: <> (> cd ./codes/paper_prediction)

[comment]: <> (> python 1_load_emb.py)

[comment]: <> (> python 2_construct_cascade.py)

[comment]: <> (> python 3_x_ids.py)

[comment]: <> (> python 4_x_idx.py)

[comment]: <> (> python 5_y.py)

[comment]: <> (```)

[comment]: <> (### Author Prediction Run scripts:)

[comment]: <> (```shell script)

[comment]: <> (> cd ./codes/paper_prediction)

[comment]: <> (> python 1_load_emb.py)

[comment]: <> (> python 2_x_y.py)

[comment]: <> (```)


[comment]: <> (## Training & evaluating SI-HDGNN)

[comment]: <> (```shell script)

[comment]: <> (> python paper_prediction.py)

[comment]: <> (> python author_prediction.py)

[comment]: <> (```)


[comment]: <> (## Options)

[comment]: <> (You may change the model settings manually in `config.py` or directly into the codes. )




[comment]: <> ([comment]: <> &#40;## Run the codes&#41;)

[comment]: <> ([comment]: <> &#40;See [README]&#40;./codes/README.md&#41; in `./codes/`.&#41;)

[comment]: <> ([comment]: <> &#40;## Todos&#41;)

[comment]: <> ([comment]: <> &#40;I plan to optimize the code in the near future, sorry for the inconvenience that recent codes are hard to read or lack of annotations.&#41;)

[comment]: <> ([comment]: <> &#40;## Cite&#41;)

[comment]: <> ([comment]: <> &#40;If you find **SI-HDGNN** useful for your research, please consider citing us 😘:&#41;)

[comment]: <> ([comment]: <> &#40;```bibtex&#41;)

[comment]: <> ([comment]: <> &#40;@inproceedings{xovee2020quantifying, &#41;)

[comment]: <> ([comment]: <> &#40;  author = {Xovee Xu and Fan Zhou and Ce Li and Goce Trajcevski and Ting Zhong and Kunpeng Zhang}, &#41;)

[comment]: <> ([comment]: <> &#40;  title = {A Heterogeneous Dynamical Graph Neural Networks Approach to Quantify Scientific Impact}, &#41;)

[comment]: <> ([comment]: <> &#40;  booktitle = {arXiv:2003.12042}, &#41;)

[comment]: <> ([comment]: <> &#40;  year = {2020}, &#41;)

[comment]: <> ([comment]: <> &#40;}&#41;)

[comment]: <> ([comment]: <> &#40;```&#41;)

[comment]: <> ([comment]: <> &#40;We also have a [survey paper]&#40;https://dl.acm.org/doi/10.1145/3433000&#41; you might be interested:&#41;)

[comment]: <> ([comment]: <> &#40;```bibtex&#41;)

[comment]: <> ([comment]: <> &#40;@article{zhou2021survey,&#41;)

[comment]: <> ([comment]: <> &#40;  author = {Fan Zhou and Xovee Xu and Goce Trajcevski and Kunpeng Zhang}, &#41;)

[comment]: <> ([comment]: <> &#40;  title = {A Survey of Information Cascade Analysis: Models, Predictions, and Recent Advances}, &#41;)

[comment]: <> ([comment]: <> &#40;  journal = {ACM Computing Surveys &#40;CSUR&#41;}, &#41;)

[comment]: <> ([comment]: <> &#40;  volume = {54},&#41;)

[comment]: <> ([comment]: <> &#40;  number = {2},&#41;)

[comment]: <> ([comment]: <> &#40;  year = {2021},&#41;)

[comment]: <> ([comment]: <> &#40;  pages = {1--36},&#41;)

[comment]: <> ([comment]: <> &#40;  doi = {10.1145/3433000},&#41;)

[comment]: <> ([comment]: <> &#40;}&#41;)

[comment]: <> ([comment]: <> &#40;```&#41;)
