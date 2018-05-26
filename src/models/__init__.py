"""Model definitions (one class per file) to define NN architectures."""
# from .example import ExampleNet

# from .vgg16 import VGG16
# from .alexnet import AlexNet
# from .mnistnet import MnistNet
# from .densenet import DenseNet

# __all__ = ('ExampleNet', 'VGG16', 'AlexNet', 'MnistNet', 'DenseNet')

# HelloWorld Ivan

from .example import ExampleNet
from .ivan import NewModel
from .cpm import Glover
__all__ = ('ExampleNet', 'NewModel', 'Glover')