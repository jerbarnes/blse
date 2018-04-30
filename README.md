Bilingual Sentiment Embeddings: Joint Projection of Sentiment Across Languages
==============

This is the finalized version of the corpora described in the following paper:

Jeremy Barnes, Roman Klinger, and Sabine Schulde im Walde. 2018. **Bilingual Sentiment Embeddings: Joint Projection of Sentiment Across Languages**. In *Proceedings of ACL 2018 (to appear)*.

The repository contains the scripts to reproduce the results from the above paper. 


If you use the code for academic research, please cite the paper in question:
```
@inproceedings{Barnes2018blse,
    author={Barnes, Jeremy and Klinger, Roman and Schulte im Walde, Sabine},
    title={Bilingual Sentiment Embeddings: Joint Projection of Sentiment Across Languages},
    booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
    year = {2018},
    month = {July},
    date = {15-20},
    address = {Melbourne, Australia},
    publisher = {Association for Computational Linguistics},
    language = {english}
    }
```


Requirements to run the experiments
--------
- Python 3
- NumPy
- sklearn [http://scikit-learn.org/stable/]
- pytorch [http://http://pytorch.org/]



Usage
--------

First, get the monolingual embeddings, either by training your own,
or by downloading the pretrained embeddings mentioned in the paper:

```
curl 
```


Then, clone the repo and run the blse script:

```
git clone https://github.com/jbarnesspain/blse
cd blse
python3 blse.py
```


License
-------

Copyright (C) 2018, Jeremy Barnes

Licensed under the terms of the Creative Commons CC-BY public license
