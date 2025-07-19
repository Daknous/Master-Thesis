import segmentation_models_pytorch as smp
from settings.config import ENCODER


def get_preprocessing_fn():
    """
    Returns the preprocessing function for the encoder backbone.
    Applies the same normalization used by the pretrained encoder.
    """
    return smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')
