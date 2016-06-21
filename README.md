overlapping categories
======================

Python 3 code for training models in a multilabel environment where categories overlap. Scripts written for "The Longue Duree of Literary Prestige" (/paceofchange repo) were subsequently revised for multilabel use in the /fiction repo (code supporting "The Life Cycles of Genres.") This is a further development of the code, starting with minor bug fixes and hopefully leading to a far-reading refactoring.

Right now the scripts use regularized logistic regression, but that's immaterial; the hard part, responsible for all the lines of code, is managing metadata. This code probably needs to be refactored to make better use of pandas; that could make it much simpler and more readable.

The first commit contains the original versions from the /fiction repo. The second commit reveals some of the bug fixes and improvements that led to this fork.
