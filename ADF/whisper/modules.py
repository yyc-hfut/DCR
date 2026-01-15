import math
from torch import nn, einsum
from einops import rearrange, repeat
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.models.clap.modeling_clap import ClapAudioLayer
from transformers.models.clap.modeling_clap import window_partition, window_reverse
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPEncoder
from timm.models.layers import trunc_normal_, DropPath


class BaseModel(nn.Module):
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def get_num_layers(self):
        return 1

    def init_extra_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        """Initialize the weights."""
        for n, m in self.named_modules():
            if 'Adapter' in n:
                for n2, m2 in m.named_modules():
                    if isinstance(m2, nn.Linear):
                        trunc_normal_(m2.weight, std=.02)
                        if m2.bias is not None:
                            nn.init.constant_(m2.bias, 0)
                    if 'D_fc2' in n2 and isinstance(m2, nn.Linear):
                        nn.init.constant_(m2.weight, 0)
                        nn.init.constant_(m2.bias, 0)
        if self.cls_head is not None:
            trunc_normal_(self.cls_head.weight, std=.02)
        if hasattr(self, 'temporal_net'):
            self.temporal_net.apply(_init_weights)


class Adapter(nn.Module):
    def __init__(self, D_features, D_hidden_features=None, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        if D_hidden_features is None:
            D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class CustomClapAudioLayer(ClapAudioLayer):
    def __init__(self, config, dim, **kwargs):
        super().__init__(config, dim, **kwargs)
        self.S_Adapter = Adapter(dim, skip_connect=True)
        self.MLP_Adapter = Adapter(dim, skip_connect=False)
        self.scale = 0.1

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not always_partition:
            self.set_shift_and_window_size(input_dimensions)
        else:
            pass
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.size()
        shortcut = hidden_states

        hidden_states = self.layernorm_before(hidden_states)

        hidden_states = hidden_states.view(batch_size, height, width, channels)

        hidden_states, pad_values = self.maybe_pad(
            hidden_states, height, width)

        _, height_pad, width_pad, _ = hidden_states.shape
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(
                hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        hidden_states_windows = window_partition(
            shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(
            -1, self.window_size * self.window_size, channels)
        attn_mask = self.get_attn_mask(
            height_pad, width_pad, dtype=hidden_states.dtype, device=hidden_states_windows.device
        )

        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        attention_output = attention_outputs[0]

        attention_windows = attention_output.view(
            -1, self.window_size, self.window_size, channels)
        shifted_windows = window_reverse(
            attention_windows, self.window_size, height_pad, width_pad)

        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:,
                                                  :height, :width, :].contiguous()

        attention_windows = attention_windows.view(
            batch_size, height * width, channels)

        hidden_states = shortcut + \
            self.S_Adapter(self.drop_path(attention_windows))

        layer_output1 = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output1)
        layer_output = hidden_states + \
            self.output(layer_output) + \
            self.drop_path(self.scale * self.MLP_Adapter((layer_output1)))

        layer_outputs = (
            layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs


class CustomWhisperEncoderLayer(WhisperEncoderLayer):
    def __init__(self, config):
        super().__init__(config)
        self.S_Adapter = Adapter(config.d_model, D_hidden_features=config.adapter_dim, skip_connect=True)
        self.MLP_Adapter = Adapter(config.d_model, D_hidden_features=config.adapter_dim, skip_connect=False)
        self.scale = config.adapter_scale

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.S_Adapter(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states2 = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states2))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states + \
            self.scale * self.MLP_Adapter(hidden_states2)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(
                hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs



class CustomCLIPEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config):
        super().__init__(config)
        self.S_Adapter = Adapter(config.hidden_size, skip_connect=True)
        self.MLP_Adapter = Adapter(config.hidden_size, skip_connect=False)
        self.scale = 0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + self.S_Adapter(hidden_states)

        residual = hidden_states
        hidden_states2 = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states2)
        hidden_states = residual + hidden_states + \
            self.scale * self.MLP_Adapter(hidden_states2)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class CustomCLIPEncoder(CLIPEncoder):
    def __init__(self, config):
        super().__init__(config)

        drop_path_rate = getattr(config, 'drop_path_rate', 0.1)  # default to 0.1
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, config.num_hidden_layers)]
        
        self.layers = nn.ModuleList([CustomCLIPEncoderLayer2(config, self.dpr[i_layer]) for i_layer in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        
class CustomCLIPEncoderLayer2(CLIPEncoderLayer):
    def __init__(self, config, drop_path=0.):
        super().__init__(config)
        self.S_Adapter = Adapter(config.hidden_size, skip_connect=True)
        self.MLP_Adapter = Adapter(config.hidden_size, skip_connect=False)
        self.scale = 0.5
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.drop_path(hidden_states)
        hidden_states = residual + self.S_Adapter(hidden_states)

        residual = hidden_states
        hidden_states2 = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states2)
        hidden_states = residual + hidden_states + self.drop_path(self.scale * self.MLP_Adapter(hidden_states2))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
    
    



class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, x2=None, **kwargs):
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x2=None, **kwargs):
        if x2 is None:
            return self.fn(self.norm(x), **kwargs)
        else:
            return self.fn(self.norm(x), self.norm(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim),
                                 GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(
            dropout)) if project_out else nn.Identity()

    def forward(self, x, x2=None):
        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        if x2 is not None:
           kv = self.to_kv(x2).chunk(2, dim=-1)

        else:
           kv = self.to_kv(x).chunk(2, dim=-1)

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                                              Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x, x2=None):
        for attn, ff in self.layers:
            x = attn(x, x2)
            x = ff(x)
        return x



class Temporal_Transformer_Cls(nn.Module):
    def __init__(self, num_patches, input_dim, depth, heads, mlp_dim, dim_head):
        super().__init__()
        dropout = 0.0
        self.num_patches = num_patches
        self.input_dim = input_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches+1, input_dim))
        self.temporal_transformer = Transformer(
            input_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :(n+1)]
        x = self.temporal_transformer(x)
        x = x[:, 0]
        return x
