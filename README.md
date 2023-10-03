# cybersecurity_IBL

Welcome—this is the backup page for my pyACT-UP (ACT-R) cybersecurity project.

**General tasks completed:**
* Get pyactup working, replicating Dr. Lebiere's LISP results, with modular structure (100% done, has all options needed)
* Get feature importance analysis working, with modular structure (100% done, has all options needed including model-based and class-based)
* Combine pureFeatureImportance project and pyactup project to run on the same model.py script
* Make randomForest output only features, rather than features/labels
* Get NN to run locally using the new file that Rob sent me
* Simplify NN, reducing it to only the needed code (50% done)

**Micro tasks TODO:**
* Remove features.csv requirerment from feature analysis code so that we can run feature analysis on datasets without a features csv, like NSL; alternatively, if this is not an option since it relies on datatypes in features.csv, add in  a features.csv file to NSL to signify having such a file is a requirement to work with this code
* Make pyactup code clearer (e.g., no need for "train_dict", use the pandas procedural datasets from the preprocessing script instead, have uniform preprocessed datasets, potentially have an original preprocessed and an epoch-wise preprocessed dataset that draws from the original?)
* Make pureFeatureImportance code cleaner, e.g., no unnecessary parameters for selectKBest, better organized/notated and more consistent functions, etc
* Get NN to run with the same datasets/parameters structure I used for actup and feature importance
* Integrate NN/pyactup/feature importance all in one folder with one parameters and one datasets file as final product

**General tasks TODO:**
* Integrate NN into pyactup
