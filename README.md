# cybersecurity_IBL

Welcome—this is the backup page for my pyACT-UP (ACT-R) cybersecurity project.

Completed:
* Get pyactup working, replicating Dr. Lebiere's LISP results, with modular structure (100% done, has all options needed)
* Get feature importance analysis working, with modular structure (100% done, has all options needed including model-based and class-based)
* Combine pureFeatureImportance project and pyactup project to run on the same model.py script
* Make randomForest output only features, rather than features/labels

**TODO:**
PART 1:
* Make pyactup code clearer (e.g., no need for "train_dict", use the pandas procedural datasets from the preprocessing script instead, have uniform preprocessed datasets, potentially have an original preprocessed and an epoch-wise preprocessed dataset that draws from the original?)
* Make pureFeatureImportance code cleaner, e.g., no unnecessary parameters for selectKBest, better organized/notated and more consistent functions, etc
* Add features csv to NSL_KDD

PART 2:
* Get NN to run locally using the new file that Rob sent me
* Get NN to run with the same datasets/parameters structure I used for actup and feature importance

PART 3:
* Integrate NN into pyactup
* Integrate NN/pyactup/feature importance all in one folder with one parameters and one datasets file as final product
