This project was developed for Statoil/C-CORE Iceberg Classifier Challenge on Kaggle

There are three main files:

1. **models.py** contains the model classes.

2. **train.py** loads and prepares training and x-validation datasets and feeds them to an ensemble of models in batches.
It uses k-folding cross validation to estimate the test error.
It also saves model checkpoints, model description, and a log for viewing in Tensorboard.
These are saved in a separate timestamped folder for each run to make it easier to keep track of experiments.

3. **test.py** loads and prepares test data. It then loads the model from the latest checkpoint created by *train.py* and iterates through the test data.
Inspired by [the work of Yarin Gal](http://mlg.eng.cam.ac.uk/yarin/blog_2248.html), I used drop-out during inference to create a confidence interval around each prediction.
This confidence interval is treated as a measure of certainty of the model in its prediction, and it is used to adjust the predictions to achieve lower log-loss.
Finally, it creates a CSV file ready to be submitted to Kaggle.

