""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Hacked together by / Copyright 2020 Ross Wightman
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.distributed import rank_zero_info
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None, relative_position_bias=None):
        B, N, C = x.shape  # B, N, 768 for base
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(
                self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = (
            qkv[0],  # B, n_head, N, 768//n_head
            qkv[1],  # ditto
            qkv[2],  # ditto
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))

        if relative_position_bias is not None:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MoEPrefixAttention(Attention):
    def __init__(
        self,
        dim,
        delta_config,  # NEW!
        with_vl,
        max_text_len,
        **kwargs
    ):
        super().__init__(dim, **kwargs)
        self.prefix_length = delta_config['prefix_length']
        self.dim = dim
        self.reparameterization = delta_config['reparameterization']
        self.max_text_len = max_text_len

        if with_vl:
            self.modality_types = {'text', 'image', 'vl'}
        else:
            self.modality_types = {'text', 'image'}

        self.prefix_indices = torch.arange(
            self.prefix_length, dtype=torch.long)

        # x2 since we directly insert tokens to K and V
        if self.reparameterization:
            # follows P-tuning v2:https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
            self.prompts = nn.ModuleDict({k: nn.Embedding(
                self.prefix_length, self.dim) for k in self.modality_types})
            # share the reparameterization projection among all modalities
            self.prompt_project = torch.nn.Sequential(
                torch.nn.Linear(dim, dim),
                torch.nn.Tanh(),
                torch.nn.Linear(dim, dim * 2),
            )
        else:
            self.prompts = nn.ModuleDict({k: nn.Embedding(
                self.prefix_length, self.dim * 2) for k in self.modality_types})

    def get_prompts(self, batch_size, modality_type):
        '''
        Input: batch(int)
        Output: Tensor(B, N_head, N_token, N_dim//N_head) * 2
        '''
        assert modality_type in {'text', 'image', 'vl'}

        def split_kv(prompts):
            key_prompts, val_prompts = torch.split(prompts, self.dim, dim=-1)
            key_prompts = key_prompts.reshape(
                batch_size, self.num_heads, self.prefix_length, self.dim // self.num_heads)
            val_prompts = val_prompts.reshape(
                batch_size, self.num_heads, self.prefix_length, self.dim // self.num_heads)
            return key_prompts, val_prompts

        if modality_type in self.modality_types:
            indices = self.prefix_indices.unsqueeze(
                0).expand(batch_size, -1).to(self.qkv.weight.device)
            prompts = self.prompts[modality_type](indices)
            if self.reparameterization:
                prompts = self.prompt_project(prompts)

            return split_kv(prompts)

        elif modality_type == 'vl':
            # concat image and text prompt for vl fusion
            indices = self.prefix_indices.unsqueeze(
                0).expand(batch_size, -1).to(self.qkv.weight.device)
            text_prompts = self.prompts['text'](
                indices)  # dim = (B, N, hidden)
            image_prompts = self.prompts['image'](indices)

            if self.reparameterization:
                text_prompts = self.prompt_project(text_prompts)
                image_prompts = self.prompt_project(image_prompts)

            return split_kv(text_prompts), split_kv(image_prompts)

    def forward(self, x, mask=None, relative_position_bias=None, modality_type=None):
        B, N, C = x.shape  # B, N, 768 for base
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(
                self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight,
                       bias=qkv_bias)  # B, N, 786*3
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = (
            qkv[0],  # B, n_head, N, 768//n_head
            qkv[1],  # ditto
            qkv[2],  # ditto
        )  # make torchscript happy (cannot use tensor as tuple)

        if modality_type in self.modality_types:
            # *Append prompt tokens
            key_prompts, val_prompts = self.get_prompts(
                B, modality_type=modality_type)
            k = torch.cat([key_prompts, k], dim=2)  # third dim is seq_len
            v = torch.cat([val_prompts, v], dim=2)

            q = q * self.scale
            attn = (q.float() @ k.float().transpose(-2, -1))

            if relative_position_bias is not None:
                # *Append zero in front of position encoding
                # attn.shape = (B, n_head, seq_len, prefix_len + seq_len); rpb.shape = (n_head, seq_len, seq_len)
                padding_shape = list(relative_position_bias.shape)
                padding_shape[-1] = self.prefix_length
                relative_position_bias = torch.cat(
                    [torch.zeros(*padding_shape, device=relative_position_bias.device), relative_position_bias], dim=-1)
                attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                # *Append True in front of mask
                mask = torch.cat(
                    [torch.ones(B, self.prefix_length, device=mask.device), mask], dim=-1)
                mask = mask.bool()  # mask.shape = B, seq_len; mask True = allow attention
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            attn = attn.softmax(dim=-1).type_as(x)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        elif modality_type == 'vl':
            # *Append both text and image prompts
            # seq = [text_prompt, text, image_prompt, image]
            (t_p_key, t_p_val), (i_p_key, i_p_val) = self.get_prompts(B, modality_type)
            # B, n_head, N, 768//n_head
            (t_key, i_key) = torch.split(
                k, [self.max_text_len, k.shape[2] - self.max_text_len], dim=2)
            (t_val, i_val) = torch.split(
                v, [self.max_text_len, v.shape[2] - self.max_text_len], dim=2)
            k = torch.cat([t_p_key,
                           t_key,
                           i_p_key,
                           i_key], dim=2)
            v = torch.cat([t_p_val,
                           t_val,
                           i_p_val,
                           i_val], dim=2)

            q = q * self.scale
            attn = (q.float() @ k.float().transpose(-2, -1))

            if relative_position_bias is not None:
                # *Append zero position encoding for prompts
                # attn.shape = (B, n_head, seq_len, prefix_len + seq_len); rpb.shape = (n_head, seq_len, seq_len)
                t_rpb, i_rpb = relative_position_bias[...,
                                                      :self.max_text_len], relative_position_bias[..., self.max_text_len:]
                padding_shape = list(relative_position_bias.shape)
                padding_shape[-1] = self.prefix_length
                padding = torch.zeros(
                    *padding_shape, device=relative_position_bias.device)
                relative_position_bias = torch.cat(
                    [padding, t_rpb, padding, i_rpb], dim=2)
                attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                # *Append True mask for prompts
                # mask.shape = B, seq_len; mask True = allow attention
                mask = mask.bool()
                t_mask, i_mask = mask[:,
                                      :self.max_text_len], mask[:, self.max_text_len:]
                padding = torch.ones(B, self.prefix_length,
                                     dtype=torch.bool, device=mask.device)
                mask = torch.cat([padding, t_mask, padding, i_mask], dim=1)
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))

            attn = attn.softmax(dim=-1).type_as(x)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        else:
            raise NotImplementedError(f'{modality_type} not implemented')


class PrefixAttention(Attention):
    def __init__(
        self,
        dim,
        delta_config,  # NEW!
        **kwargs
    ):
        super().__init__(dim, **kwargs)
        self.prefix_length = delta_config['prefix_length']
        self.dim = dim
        self.reparameterization = delta_config['reparameterization']
        self.prefix_indices = torch.arange(
            self.prefix_length, dtype=torch.long)

        # x2 since we directly insert tokens to K and V
        if self.reparameterization:
            # follows P-tuning v2:https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
            self.prompts = nn.Embedding(self.prefix_length, self.dim)
            self.prompt_project = torch.nn.Sequential(
                torch.nn.Linear(dim, dim),
                torch.nn.Tanh(),
                torch.nn.Linear(dim, dim * 2),
            )
        else:
            self.prompts = nn.Embedding(self.prefix_length, self.dim * 2)

    def get_prompts(self, batch_size):
        '''
        Input: batch(int)
        Output: Tensor(B, N_head, N_token, N_dim//N_head) * 2
        '''
        indices = self.prefix_indices.unsqueeze(
            0).expand(batch_size, -1).to(self.qkv.weight.device)
        prompts = self.prompts(indices)
        if self.reparameterization:
            prompts = self.prompt_project(prompts)
        key_prompts, val_prompts = torch.split(prompts, self.dim, dim=-1)
        key_prompts = key_prompts.reshape(batch_size, self.num_heads,
                                          self.prefix_length, -1)
        val_prompts = val_prompts.reshape(batch_size, self.num_heads,
                                          self.prefix_length, -1)
        return key_prompts, val_prompts

    def forward(self, x, mask=None, relative_position_bias=None):
        B, N, C = x.shape  # B, N, 768 for base
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(
                self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight,
                       bias=qkv_bias)  # B, N, 786*3
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = (
            qkv[0],  # B, n_head, N, 768//n_head
            qkv[1],  # ditto
            qkv[2],  # ditto
        )  # make torchscript happy (cannot use tensor as tuple)

        # *Append prompt tokens
        key_prompts, val_prompts = self.get_prompts(B)
        k = torch.cat([key_prompts, k], dim=2)  # third dim is seq_len
        v = torch.cat([val_prompts, v], dim=2)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))

        if relative_position_bias is not None:
            # *Append zero in front of position encoding
            # attn.shape = (B, n_head, seq_len, prefix_len + seq_len); rpb.shape = (n_head, seq_len, seq_len)
            padding_shape = list(relative_position_bias.shape)
            padding_shape[-1] = self.prefix_length
            relative_position_bias = torch.cat(
                [torch.zeros(*padding_shape, device=relative_position_bias.device), relative_position_bias], dim=-1)
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # *Append True in front of mask
            mask = torch.cat(
                [torch.ones(B, self.prefix_length, device=mask.device), mask], dim=-1)
            mask = mask.bool()  # mask.shape = B, seq_len; mask True = allow attention
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        with_vlffn=False,
        layer_scale_init_values=0.1,
        max_text_len=40,
        delta_config=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if delta_config is None or delta_config['type'] is None:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif delta_config['type'] == 'bitfit':
            pass  # handled in run.py
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif delta_config['type'] == 'prefix':
            self.attn = PrefixAttention(
                dim,
                delta_config,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif delta_config['type'] == 'moe_prefix':
            self.attn = MoEPrefixAttention(
                dim,
                delta_config,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                with_vl=with_vlffn,
                max_text_len=max_text_len,
            )
        else:
            raise Exception(f'wrong delta config: {delta_config}')

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2_text = norm_layer(dim)
        self.norm2_imag = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_text = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp_imag = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp_vl = None
        if with_vlffn:
            self.mlp_vl = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
            self.norm2_vl = norm_layer(dim)

        self.gamma_1 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)), requires_grad=True) \
            if layer_scale_init_values is not None else 1.0
        self.gamma_2 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)), requires_grad=True) \
            if layer_scale_init_values is not None else 1.0

        self.max_text_len = max_text_len

    def forward(self, x, mask=None, modality_type=None, relative_position_bias=None):
        if isinstance(self.attn, MoEPrefixAttention):
            x = x + self.drop_path(
                self.gamma_1 *
                self.attn(
                    self.norm1(x),
                    mask=mask, relative_position_bias=relative_position_bias, modality_type=modality_type,
                )
            )

        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x),
                                                            mask=mask, relative_position_bias=relative_position_bias))

        if modality_type == "image":
            x = x + self.drop_path(self.gamma_2 *
                                   self.mlp_imag(self.norm2_imag(x)))
        elif modality_type == "text":
            x = x + self.drop_path(self.gamma_2 *
                                   self.mlp_text(self.norm2_text(x)))
        else:
            if self.mlp_vl is None:
                x_text = x[:, : self.max_text_len]
                x_imag = x[:, self.max_text_len:]
                x_text = x_text + \
                    self.drop_path(
                        self.gamma_2 * self.mlp_text(self.norm2_text(x_text)))
                x_imag = x_imag + \
                    self.drop_path(
                        self.gamma_2 * self.mlp_imag(self.norm2_imag(x_imag)))
                x = torch.cat([x_text, x_imag], dim=1)
            else:
                x = x + self.drop_path(self.gamma_2 *
                                       self.mlp_vl(self.norm2_vl(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        no_patch_embed_bias=False,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.patch_shape = (
            img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False if no_patch_embed_bias else True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # FIXME look at relaxing size constraints
        x = self.proj(x)
        return x


class MultiWayTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        need_relative_position_embed=True,
        use_abs_pos_emb=False,
        layer_scale_init_values=0.1,
        vlffn_start_layer_index=10,
        config=None,
    ):
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
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            need_relative_position_embed (bool): enable relative position bias on self-attention
            use_abs_pos_emb (bool): enable abs pos emb
            layer_scale_init_values (float or None): layer scale init values, set None to disable
            vlffn_start_layer_index (int): vl-ffn start index
            config: (dict): other hyper from pytorch-lighting
        """
        super().__init__()
        self.delta_config = config['delta']
        drop_path_rate = drop_path_rate if config is None else config["drop_path_rate"]
        rank_zero_info("drop path rate: {}".format(drop_path_rate))
        self.use_abs_pos_emb = use_abs_pos_emb
        self.need_relative_position_embed = need_relative_position_embed

        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.vlffn_start_layer_index = vlffn_start_layer_index
        if config["loss_names"]["textmlm"] > 0:
            self.vlffn_start_layer_index = depth
            rank_zero_info(
                "Set vlffn_start_layer_index={} for text-only pretraining".format(self.vlffn_start_layer_index))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + 1, embed_dim)) if self.use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    with_vlffn=(i >= self.vlffn_start_layer_index),
                    layer_scale_init_values=layer_scale_init_values,
                    max_text_len=config["max_text_len"],
                    delta_config=self.delta_config,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @ torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def visual_embed(self, _x):
        x = self.patch_embed(_x)
        x = x.flatten(2).transpose(1, 2)
        B, L, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        x_mask = torch.ones(x.shape[0], x.shape[1])

        return x, x_mask


# VLMo base/p16
@ register_model
def vlmo_base_patch16(pretrained=False, **kwargs):
    img_size = kwargs.pop("img_size", 224)
    model = MultiWayTransformer(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True, vlffn_start_layer_index=10,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# VLMo large/p16


@ register_model
def vlmo_large_patch16(pretrained=False, **kwargs):
    img_size = kwargs.pop("img_size", 224)
    model = MultiWayTransformer(
        img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True, vlffn_start_layer_index=21,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# VLMo base+/p16


@ register_model
def vlmo_base_plus_patch16(pretrained=False, **kwargs):
    img_size = kwargs.pop("img_size", 224)
    model = MultiWayTransformer(
        img_size=img_size, patch_size=16, embed_dim=544, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True, vlffn_start_layer_index=21,
        use_abs_pos_emb=True, need_relative_position_embed=False,
        layer_scale_init_values=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
