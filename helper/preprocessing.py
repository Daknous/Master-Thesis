import segmentation_models_pytorch as smp

# Encoder name must match the one used in models.py
ENCODER = 'resnet34'


def get_preprocessing_fn():
    """
    Returns the preprocessing function for the encoder backbone.
    Applies the same normalization used by the pretrained encoder.
    """
    return smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')
