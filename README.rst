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
* Documentation: https://crossact.readthedocs.io

========
Datasets
========
* `Full Dataset <https://drive.google.com/open?id=14GPvxHjoC3A1nmLNJNlTSSI_hiBU9jGw>`_
* `Toy Dataset (100users) <https://drive.google.com/open?id=1NM4g0oJ8O5yxc980qR2j-UhJ4miSpaIG>`_

========
Code
========
* KEEN MODEL
	* `Running the code to save the data <https://github.com/oentaryorj/crossact/blob/master/james_code/load_data_ver2.py>`_
		* We need to create three files for each Github or StackOverflow dataset (i.e., interactions_non_activities.pickle, i_features_non_activities.pickle, and u_features_CO.pickle)
			* interactions_non_activities.pickle: interaction between users and items
			* i_features_non_activities.pickle: item features
			* u_features_CO.pickle: user features 
	* `Training KEEN model <https://github.com/oentaryorj/crossact/blob/master/james_code/keen2act_training_ver1.py>`_
		* We have two steps to train KEEN model:
			* Training interaction model (See step 1a in the KEEN model)
			* Training threshold model (See step 1b in the KEEN model)

* ACT MODEL
	* `Running the code to save the data <https://github.com/oentaryorj/crossact/blob/master/james_code/load_data_ver2.py>`_
		* We need to create two files for each Github or StackOverflow dataset (i.e., interactions_with_first_activities_for_ACT.pickle, interactions_with_second_activities_for_ACT.pickle)
			* interactions_with_first_activities_for_ACT.pickle: interaction between users and items for the first activity
			* interactions_with_second_activities_for_ACT.pickle: interaction between users and items for the second activity
	* `Training ACT model <https://github.com/oentaryorj/crossact/blob/master/james_code/keen2act_training_ver1.py>`_
		* We have two steps to train ACT model:
			* Training interaction model (See in the ACT model)
			* Training threshold model (See in the ACT model)			

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage


