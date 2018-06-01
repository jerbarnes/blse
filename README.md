Bilingual Sentiment Embeddings: Joint Projection of Sentiment Across Languages
==============

This is the source code from the ACL paper:

Jeremy Barnes, Roman Klinger, and Sabine Schulde im Walde. 2018. **Bilingual Sentiment Embeddings: Joint Projection of Sentiment Across Languages**. In *Proceedings of ACL 2018 (to appear)*.


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
- pytorch [http://pytorch.org/]



Usage
--------

First, clone the repo:

```
git clone https://github.com/jbarnesspain/blse
cd blse
```


Then, get monolingual embeddings, either by training your own,
or by downloading the [pretrained embeddings](https://drive.google.com/open?id=1GpyF2h0j8K5TKT7y7Aj0OyPgpFc8pMNS) mentioned in the paper,
and put them in the 'embeddings' directory:


Run the blse parameter search script, in order to get the best hyperparameters:

```
python3 parmeter_search.py
```

Finally, you can use the blse.py script which will automatically use the best hyperparameters found.

```
python3 blse.py
``` 


License
-------

Copyright (C) 2018, Jeremy Barnes

Licensed under the terms of the Creative Commons CC-BY public license
