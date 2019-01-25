========
crossact
========


.. image:: https://img.shields.io/pypi/v/crossact.svg
        :target: https://pypi.python.org/pypi/crossact

.. image:: https://img.shields.io/travis/oentaryorj/crossact.svg
        :target: https://travis-ci.org/oentaryorj/crossact

.. image:: https://readthedocs.org/projects/crossact/badge/?version=latest
        :target: https://crossact.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Python Boilerplate contains all the boilerplate you need to create a Python package.


* Free software: BSD license
* Documentation: https://crossact.readthedocs.io.


Factoid Embedding
-----------------

Example to run the codes
^^^^^^^^^^^^^^^^^^^^^^^^

::
   python main.py --path ../data_for_experiment/facebook_twitter/ \
                   --source_prefix fb --target_prefix tw --source_col 0 --target_col 1  \
                   --name_method tfidf --image_exist True --user_dim 1024+256 \
                   --name_concatenate True \
                   --image_dim 256 --n_iter 52 --supervised False\
                    --image_method vgg16 --skip_network False;


Usage
^^^^^

::
   usage: main.py [-h] [--path PATH] [--skip_network SKIP_NETWORK]
                  [--source_prefix SOURCE_PREFIX] [--target_prefix TARGET_PREFIX]
                  [--source_col SOURCE_COL] [--target_col TARGET_COL]
                  [--name_dim NAME_DIM] [--name_concatenate NAME_CONCATENATE]
                  [--name_preprocess NAME_PREPROCESS] [--name_method NAME_METHOD]
                  [--screen_name_exist SCREEN_NAME_EXIST]
                  [--image_exist IMAGE_EXIST] [--image_method IMAGE_METHOD]
                  [--image_identical_threshold IMAGE_IDENTICAL_THRESHOLD]
                  [--image_dim IMAGE_DIM]
                  [--cosine_embedding_batch_size COSINE_EMBEDDING_BATCH_SIZE]
                  [--cosine_embedding_learning_rate COSINE_EMBEDDING_LEARNING_RATE]
                  [--supervised SUPERVISED] [--snapshot SNAPSHOT]
                  [--snapshot_gap SNAPSHOT_GAP] [--n_iter N_ITER]
                  [--warm_up_iter WARM_UP_ITER] [--user_dim USER_DIM]
                  [--nce_sampling NCE_SAMPLING]
                  [--triplet_embedding_batch_size TRIPLET_EMBEDDING_BATCH_SIZE]
                  [--triplet_embedding_learning_rate_f TRIPLET_EMBEDDING_LEARNING_RATE_F]
                  [--triplet_embedding_learning_rate_a TRIPLET_EMBEDDING_LEARNING_RATE_A]
                  [--stratified_attribute STRATIFIED_ATTRIBUTE]

Reference
^^^^^^^^^

.. [#] Wei Xie, Xin Mu, Roy Ka-Wei Lee, Feida Zhu, and Ee-Peng Lim, "Unsupervised User Identity Linkage via Factoid Embedding",
*Proceedings of the IEEE International Conference on Data Mining (ICDM'18)*, Singapore [Paper_]

.. _Paper: https://arxiv.org/pdf/1901.06648.pdf

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
