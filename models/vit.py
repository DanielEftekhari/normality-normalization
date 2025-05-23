# adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py

import collections
from itertools import repeat
from collections import OrderedDict
import math
import torch
import torch.nn as nn

import layers as norm_layers


class MLP(torch.nn.Sequential):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        norm_layer=None,
        activation_layer=torch.nn.ReLU,
        bias=True,
        dropout=0.0,
        device=None,
        *args, **kwargs
        ):

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim, device=device, **kwargs))
            layers.append(activation_layer())
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout))

        super().__init__(*layers)


class MLPBlock(MLP):
    _version = 2

    def __init__(self,
                 in_dim,
                 mlp_dim,
                 dropout
                 ):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout,
        attention_dropout,
        norm_layer='normal',
        activation='gelu',
        cfg=None,
        device=None,
        *args, **kwargs
        ):
        super().__init__()

        self._norm_layer = norm_layer
        if norm_layer == 'normal': self.norm_layer = norm_layers.LayerNormalNorm
        elif norm_layer == 'layer': self.norm_layer = norm_layers.LayerNorm
        elif norm_layer == 'none': self.norm_layer = nn.Identity

        self.num_heads = num_heads

        # Attention block
        self.norm_1 = self.norm_layer(hidden_dim, device=device, **kwargs)
        self.self_attention = nn.MultiheadAttention(hidden_dim,
                                                    num_heads,
                                                    dropout=attention_dropout,
                                                    batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.norm_2 = self.norm_layer(hidden_dim, device=device, **kwargs)
        self.mlp = MLPBlock(hidden_dim,
                            mlp_dim,
                            dropout)

    def forward(self, input):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.norm_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.norm_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    def __init__(
        self,
        seq_length,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout,
        attention_dropout,
        norm_layer='normal',
        activation='gelu',
        cfg=None,
        device=None,
        *args, **kwargs
        ):
        super().__init__()

        self._norm_layer = norm_layer
        if norm_layer == 'normal': self.norm_layer = norm_layers.LayerNormalNorm
        elif norm_layer == 'layer': self.norm_layer = norm_layers.LayerNorm
        elif norm_layer == 'none': self.norm_layer = nn.Identity

        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer=norm_layer,
                activation=activation,
                cfg=cfg,
                device=device,
                **kwargs
            )
        self.layers = nn.Sequential(layers)
        self.norm = self.norm_layer(hidden_dim, device=device, **kwargs)

    def forward(self, input):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.norm(self.layers(self.dropout(input)))


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
                self,
                image_size,
                patch_size,
                num_layers,
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout=0.0,
                attention_dropout=0.0,
                num_classes=1000,
                representation_size=None,
                c_in=3,
                standardize=None,
                norm_layer='normal',
                activation='gelu',
                cfg=None,
                device=None,
                *args, **kwargs
                ):
        super().__init__()

        self.c_in = c_in

        if standardize is not None:
            self.standardize = norm_layers.Standardize(mean=standardize[0], std=standardize[1], device=device)
        else:
            self.standardize = nn.Identity()

        self._norm_layer = norm_layer
        if norm_layer == 'normal': self.norm_layer = norm_layers.LayerNormalNorm
        elif norm_layer == 'layer': self.norm_layer = norm_layers.LayerNorm
        elif norm_layer == 'none': self.norm_layer = nn.Identity

        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size

        self.conv_proj = nn.Conv2d(
                in_channels=self.c_in, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer=norm_layer,
            activation=activation,
            cfg=cfg,
            device=device,
            **kwargs
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x):
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x):
        x = self.standardize(x)

        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x


def _vision_transformer(
                    patch_size,
                    num_layers,
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    **kwargs,
                    ):
    image_size = kwargs.pop("image_size", 224)

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )
    return model


def vit_b_16(**kwargs):
    return _vision_transformer(
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            **kwargs,
            )


def _make_ntuple(x, n):
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise, we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8

    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))
