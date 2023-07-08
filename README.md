# cybersecurity_IBL

Welcome—this is the backup page for my local save of the pyACTUp cybersecurity project.

In order to set up the model, adjust the parameters in `model.py` to your liking and run it using `py model.py`.  The ULRs should link to CSV files in the NSL_KDD dataset (https://github.com/defcom17/NSL_KDD/tree/master), which are used for training and testing.

The current state of the project is the following:

* Training is presumably fine (because retrieval has hits)
* Retrieval:
    * No differences between partial and strict matching
    * Retrieval with no partial matching is fine:
        * When the model is trained on the entire training set (1~4% accuracy range, as expected)
        * When the model is trained on a subset of the training set (0% accuracy, as expected)
        * When the model retrieves from the training set, rather than from the separate testing set (100% accuracy, as expected)
    * Retrieval with partial matching:
        * Not fine when trained on the entire training set (1~4% accuracy range, not expected)
        * Not fine when trained on a subset of the training set (0% accuracy, not expected)
        * Fine when the model retrieves from the training set, rather than from the separate testing set (100% accuracy, as expected)
* Blending (using the best_blend function):
    * When it makes a prediction other than 'None': `RuntimeError: Error computing blended value, is perhaps the value of the threat category slotis  not numeric in one of the matching chunks? (could not convert string to float: 'smurf')`
