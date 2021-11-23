import torch
import torch.nn as nn
from functools import partial
import torch.nn.init as init


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384'
]


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.in_channels = in_channels
        self.embed_dim   = embed_dim

        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels=self.in_channels, out_channels=self.embed_dim,
                                        kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        return self.patch_embed(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, qkv_bias=False, att_drop_rate=0.0, proj_drop_rate=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = self.embed_dim // self.num_heads

        self.qkv        = nn.Linear(in_features=embed_dim, out_features=embed_dim * 3, bias=qkv_bias)
        self.atten_drop = nn.Dropout(p=att_drop_rate)
        self.proj       = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.proj_drop  = nn.Dropout(p=proj_drop_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        k = k.transpose(-2, -1)

        att = self.atten_drop((torch.matmul(q, k) * (self.head_dim ** -0.5)).softmax(dim=-1))
        att = self.proj_drop(self.proj(torch.matmul(att, v).transpose(1, 2).reshape(B, N, C)))

        return att


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop_rate=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_path_rate=0.0):
        super().__init__()

        self.drop_path_rate = drop_path_rate
        self.keep_prob = 1 - self.drop_path_rate

    def forward(self, x):
        if not self.training or self.drop_path_rate == 0.0:
            return x

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = (self.keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)).floor()
        x = x.div(self.keep_prob) * random_tensor

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, qkv_bias=False, mlp_ratio=4, drop_rate=0.0,
                    att_drop_rate=0.0, drop_path_rate=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.att = Attention(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                            att_drop_rate=att_drop_rate, proj_drop_rate=drop_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = norm_layer(embed_dim)
        self.mlp = MLP(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), act_layer=act_layer, drop_rate=drop_rate)

    def forward(self, x):
        x = x + self.drop_path(self.att(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class VisionTransformer(nn.Module):
    """
    Based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU, weight_init=''):
        super().__init__()
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """

        self.num_classes = num_classes
        self.embed_dim   = embed_dim
        self.num_tokens  = 2 if distilled else 1

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            TransformerEncoder(
                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
                att_drop_rate=attn_drop_rate, drop_path_rate=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)


        # Classifier head(s)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        init.trunc_normal_(self.pos_embed, std=.02)
        init.trunc_normal_(self.cls_token, std=.02)
        if self.dist_token is not None:
            init.trunc_normal_(self.dist_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):  # Add for deit
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            dist_token = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.dist_token is None:
            return x[:, 0]
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)

        if self.head_dist is None:
            return self.head(x)
        else:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training:
                return x, x_dist
            else:
                return (x + x_dist) / 2


def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, distilled=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, distilled=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, distilled=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, distilled=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
