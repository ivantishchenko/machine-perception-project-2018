# Hand joint Estimation Skeleton Code
Visit [here](https://ait.ethz.ch/teaching/courses/2018-SS-Machine-Perception/) for more information about the Machine Perception course.

All questions should first be directed to [our course Piazza](https://piazza.com/class/jdbpmonr7fa26b) before being sent to my [e-mail address](mailto:adrian.spurr@inf.ethz.ch). This skeleton code was developed by Seonwook Park and has been adapted for this project.

## Setup

The following two steps will prepare your environment to begin training and evaluating models.

### Downloading necessary datasets

Download the .h5 file from the Kaggle page

### Installing dependencies

Run (with `sudo` appended if necessary),
```
python3 setup.py install
```

Note that this can be done within a [virtual environment](https://docs.python.org/3/tutorial/venv.html). In this case, the sequence of commands would be similar to:
```
mkvirtualenv -p $(which python3) myenv
python3 setup.py install
```

## Structure

* `datasets/` - all data sources required for training/validation/testing.
* `outputs/` - any output for a model will be placed here, including logs, summaries, checkpoints, and Kaggle submission `.csv` files.
* `src/` - all source code.
    * `core/` - base classes
    * `datasources/` - routines for reading and preprocessing entries for training and testing
    * `models/` - neural network definitions
    * `util/` - utility methods
    * `main.py` - training script

## Creating your own model
### Model definition
To create your own neural network, do the following:
1. Make a copy of `src/models/example.py`. For the purpose of this documentation, let's call the new file `newmodel.py` and the class within `NewModel`.
2. Now edit `src/models/__init__.py` and insert the new model by making it look like:
```
from .example import ExampleNet
from .newmodel import NewModel
__all__ = ('ExampleNet', 'NewModel')
```
3. Lastly, make a copy or edit `src/main.py` such that it imports and uses class `NewModel` instead of `ExampleNet`.

### Training the model
If your training script is called `main.py`, simply `cd` into the `src/` directory and run
```
python3 main.py
```

### Outputs
When your model has completed training, it will perform a full evaluation on the test set. For class `ExampleNet`, this output can be found in the folder `outputs/ExampleNet/` as `to_submit_to_kaggle_XXXXXXXXX.csv`.

Submit this `csv` file to our page on [Kaggle](https://www.kaggle.com/c/mp18-hand-joint-recognition/submissions).
