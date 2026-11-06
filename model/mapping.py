from __future__ import annotations

"""Per-domain spectral mapping: 1x1 convolution followed by batch norm."""

from torch import nn


class Mapping(nn.Module):
    """Map domain-specific spectral bands into the shared input dimension."""

    def __init__(self, in_dimension: int, out_dimension: int):
        super().__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        return self.preconv_bn(self.preconv(x))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Older checkpoints stored the same layers as Sequential indices 0 and 1.
        for old, new in (("0.", "preconv."), ("1.", "preconv_bn.")):
            for key in list(state_dict):
                if key.startswith(prefix + old):
                    new_key = prefix + new + key[len(prefix + old):]
                    if new_key not in state_dict:
                        state_dict[new_key] = state_dict[key]
                    del state_dict[key]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
