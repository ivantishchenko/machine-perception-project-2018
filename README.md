# Hand Joint Estimation &mdash; Machine Perception Project 2018
This repository summarises our findings and models we used for the task of estimating hand joint locations.

All questions should be directed towards the authors in case discrepancies are found or our README is incomplete in
instructions.

The skeleton code was developed by Seonwook Park, modified for hand joint estimation by Adrian Spurr and further adapted
to our needs.

## Some Predictions on the Testing Data

### Inception-ResNet-v2
![Incetion-ResNet-v2 Image 1](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/incres0.png "Inception-ResNet-v2 Prediction 1")
![Incetion-ResNet-v2 Image 2](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/incres1.png "Inception-ResNet-v2 Prediction 2")
![Incetion-ResNet-v2 Image 3](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/incres2.png "Inception-ResNet-v2 Prediction 3")
![Incetion-ResNet-v2 Image 4](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/incres3.png "Inception-ResNet-v2 Prediction 4")
![Incetion-ResNet-v2 Image 5](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/incres4.png "Inception-ResNet-v2 Prediction 5")

### CPM
![CPM Image 1](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/cpm0.png "CPM Prediction 1")
![CPM Image 2](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/cpm1.png "CPM Prediction 2")
![CPM Image 3](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/cpm2.png "CPM Prediction 3")
![CPM Image 4](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/cpm3.png "CPM Prediction 4")
![CPM Image 5](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/cpm4.png "CPM Prediction 5")

### ResNet34
![ResNet34 Image 1](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/resnet0.png "ResNet34 Prediction 1")
![ResNet34 Image 2](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/resnet1.png "ResNet34 Prediction 2")
![ResNet34 Image 3](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/resnet2.png "ResNet34 Prediction 3")
![ResNet34 Image 4](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/resnet3.png "ResNet34 Prediction 4")
![ResNet34 Image 5](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/resnet4.png "ResNet34 Prediction 5")

### Inception-v3
![Incetion-v3 Image 1](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/incep0.png "Inception-v3 Prediction 1")
![Incetion-v3 Image 2](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/incep1.png "Inception-v3 Prediction 2")
![Incetion-v3 Image 3](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/incep2.png "Inception-v3 Prediction 3")
![Incetion-v3 Image 4](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/incep3.png "Inception-v3 Prediction 4")
![Incetion-v3 Image 5](https://github.com/ivantishchenko/machine-perception-project-2018/raw/master/plots/img/sub/incep4.png "Inception-v3 Prediction 5")


## Setup
The following steps need to be followed to train our networks.

### Download Necessary Dataset
Download the .h5 files for training and testing from the Kaggle page. Be mindful to download the training set with
keypoint visibility indicators and name it `training.h5`.

### Installing Dependencies
Run (with `sudo` appended if necessary),
```
python3 setup.py install
```

Note that this can be done within a [virtual environment](https://docs.python.org/3/tutorial/venv.html). In this case,
the sequence of commands would be similar to:
```
mkvirtualenv -p $(which python3) myenv
python3 setup.py install
```

We additionally need [this library](https://github.com/aleju/imgaug) installed for python3. If the previous step does
not list it as installed using
```
pip3 freeze | grep imgaug
```
then installation instructions should be followed and
guidelines of your distribution followed. We used version 0.2.5.

On Debian 9 the following command worked in a virtualenv without problems:
```
pip install git+https://github.com/aleju/imgaug
```

### Create Training and Validation Splits
Run as a normal user
```
cd datasets/
python3 ./train_validate_split.py
```

This generates a new file `dataset.h5` which has 48856 training samples and 8610 validation samples, reachable with the
keys `train` and `validate`.

## Running Networks
We have the following selection of networks for quick training:
  * Inception-ResNet-v2 (`inception-resnet`)
  * ResNet34 (`dressnet`)
  * CPM (`cpm`)
  * Inception-v3 (`inception`)

All networks are called with their own `main-*.py` file in `./src`. Their source code can be found in `./src/models/*`

As we experimented with different training pipelines (mainly image augmentation) we prepared git commits which reflect
the whole state of the repository. Checkout the repository to one of the following commits execute then:
```
cd src
python3 ./main-$NETWORK.py
```
where `$NETWORK` is one of the four types as explained above.

The commit commits are as follows (please update to current HEAD for a full list):
  * Inception-ResNet-v2: 915ae3f30ca04ecac046fb58cb04d2b372e9bde0
  * ResNet34: 8326e1e8aa0fbe48d078e15674c99e2724432615
  * CPM: 8326e1e8aa0fbe48d078e15674c99e2724432615
  * Inception-v3: 915ae3f30ca04ecac046fb58cb04d2b372e9bde0

## Pretrained Models
We have decided to further make available the final models of all four networks: You may access them at this link:
[]

## Report
The report including source files is in `./report`. We publish this work using the Attribution-NonCommercial-ShareAlike
(CC BY-NC-SA) license.

## Code License
See `LICENSE`.


# General Information of the Project Template
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
