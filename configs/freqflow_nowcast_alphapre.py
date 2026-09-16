"""Full-capacity FreqFlow configuration for VAE-space Meteo nowcasting.

The Transformer/ConvNeXt widths, depths, heads, hidden dimensions, kernels,
normalization and dropout are retained from ``configs/freqflow.py``.  Five
observed and twenty future radar frames each carry four VAE channels, so time
and latent channels are flattened to 20 conditioning and 80 generated channels.
"""

from dataclasses import dataclass


@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


model = Args(
    channels=80,
    history_channels=20,
    history_feature_dim=64,
    block_grad_to_lowres=False,
    norm_type="TDRMSN",
    stage_configs=[
        Args(
            block_type="TransformerBlock",
            dim=960,
            hidden_dim=1920,
            num_attention_heads=16,
            num_blocks=39,
            # Match the original FreqFlow latent geometry: 32px -> 16px.
            max_height=16,
            max_width=16,
            image_input_ratio=1,
            input_feature_ratio=2,
            final_kernel_size=3,
            dropout_prob=0,
        ),
        Args(
            block_type="ConvNeXtBlock",
            dim=480,
            hidden_dim=960,
            kernel_size=7,
            num_blocks=20,
            max_height=32,
            max_width=32,
            image_input_ratio=1,
            input_feature_ratio=1,
            final_kernel_size=3,
            dropout_prob=0,
        ),
    ],
)


def get_config():
    return Args(
        radar_img_size=128,
        latent_size=32,
        latent_channels=4,
        frames_in=5,
        frames_out=20,
        nnet=Args(name="freqflow", model_args=model),
    )
