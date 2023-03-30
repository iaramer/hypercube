import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder, base_architecture
from torch.nn import LayerNorm
import torch.nn.functional as F
from hc_multihead_attention import HCMultiheadAttention
from hc import HyperCubeBlock


# TODO: add checks for valid HyperCubeBlock dims
class HCTransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = HCMultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(activation=args.activation_fn)
        self.activation_dropout = args.activation_dropout

        self.fc1 = HyperCubeBlock(self.embed_dim, args.encoder_ffn_embed_dim) 
        self.fc2 = HyperCubeBlock(args.encoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask, attn_mask=None):
        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn


# TODO: continue
class HCTransformerDecoderLayer(nn.Module):
    def __init__(self, args, *extra_args, **extra_kwargs):
        super().__init__()
        self.self_attn = ...
        self.fc1 = HCLayer(...)  # Replace this with the correct initialization for your HCLayer
        self.fc2 = HCLayer(...)  # Replace this with the correct initialization for your HCLayer
        ...


class HCTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            HCTransformerEncoderLayer(args) for _ in range(args.encoder_layers)
        ])


class HCTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            HCTransformerDecoderLayer(args) for _ in range(args.decoder_layers)
        ])


class HCTransformerModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return HCTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return HCTransformerDecoder(args, tgt_dict, embed_tokens)


register_model('hc_transformer', HCTransformerModel)

def hc_base_architecture(args):
    base_architecture(args)

register_model_architecture('hc_transformer', 'hc_transformer_base', hc_base_architecture)
