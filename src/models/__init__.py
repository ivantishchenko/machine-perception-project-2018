"""Model definitions (one class per file) to define NN architectures."""
# from .example import ExampleNet

# from .vgg16 import VGG16
# from .alexnet import AlexNet
# from .mnistnet import MnistNet
# from .densenet import DenseNet

# __all__ = ('ExampleNet', 'VGG16', 'AlexNet', 'MnistNet', 'DenseNet')

from .example import ExampleNet
from .cpm import Glover
from .dressnet import ResNet
from .inceptionresnet import IncResNet
__all__ = ('ExampleNet', 'Glover', 'ResNet', 'IncResNet')
