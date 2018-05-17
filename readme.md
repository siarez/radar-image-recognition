This project was developed for Statoil/C-CORE Iceberg Classifier Challenge on Kaggle

There are three main files:

1. **models.py** contains the model classes.

2. **train.py** loads and prepares training and x-validation datasets and feeds them to an ensemble of models in batches.
It uses k-folding cross validation to estimate the test error.
It also saves model checkpoints, model description, and a log for viewing in Tensorboard.
These are saved in a separate timestamped folder for each run to make it easier to keep track of experiments.

3. **test.py** loads and prepares test data. It then loads the model from the latest checkpoint created by *train.py* and iterates through the test data.
Finally, it creates a CSV file ready to be submitted to Kaggle.

