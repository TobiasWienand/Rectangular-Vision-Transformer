# Rectangular Vision Transformer (ViT)

This repository contains an implementation of the Vision Transformer (ViT) model that supports rectangular patches. The original implementation can be found at [vit-pytorch](https://github.com/lucidrains/vit-pytorch). Our implementation extends the original model to handle non-square patches, which can be useful for specific use cases such as time-frequency analysis.

## Use Case: Time-Frequency Analysis

In time-frequency analysis, the time and frequency resolutions are often unequal. For example, in spectrograms or wavelet transforms, the frequency axis may have a finer resolution than the time axis. When using a Vision Transformer for processing such data, it may be necessary to use rectangular patches that can better capture the inherent structure of the time-frequency representation.

Rectangular patches can help the model capture the variations in the time and frequency dimensions more effectively, potentially leading to better performance in tasks such as audio classification, speech recognition, or anomaly detection in time series data.

## Example Usage

To use the Rectangular ViT, first import the required classes and functions:

```python
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from rectangular_vit import ViT  # Assuming you saved the modified ViT implementation in a file named rectangular_vit.py
```

Then, create an instance of the Rectangular ViT model with desired parameters:

```python
model = ViT(
    image_size=(64, 64),  # Image height and width
    patch_height=16,  # Patch height
    patch_width=8,  # Patch width
    num_classes=10,  # Number of output classes
    dim=512,  # Transformer dimension
    depth=6,  # Number of layers in the transformer
    heads=8,  # Number of heads in multi-head self-attention
    mlp_dim=2048,  # Hidden dimension of the feedforward layer
    channels=3,  # Number of input channels
    dim_head=64,  # Dimension of each head in multi-head self-attention
    dropout=0.1,  # Dropout rate
    emb_dropout=0.1  # Embedding dropout rate
)
```

Now you can use the model object for training or inference on your time-frequency data.

## Credit
This implementation is based on the vit-pytorch repository by Phil Wang (lucidrains). We would like to express our gratitude for their work on the original implementation, which served as a starting point for the Rectangular ViT model.

## Contact
If this code doesn't work for you, contact tobias.wienand@rub.de