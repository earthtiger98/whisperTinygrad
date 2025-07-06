"""
Complete tinygrad implementation of Whisper model
Based on tinygrad/examples/whisper.py with full functionality
"""
import base64
import gzip
import collections
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Union, Literal, List

import numpy as np
from tinygrad import Tensor, TinyJit, Variable, nn
from tinygrad.nn.state import torch_load, load_state_dict
from tinygrad.helpers import getenv, fetch

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = np.exp(-log_timescale_increment * np.arange(channels // 2))
    scaled_time = np.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)


class MultiHeadAttention:
    def __init__(self, n_state, n_head, kv_caching: Literal['cross', 'self']=None, max_self_attn_cache_len=None):
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

        self.kv_caching = kv_caching
        self.max_self_attn_cache_len = max_self_attn_cache_len

    def __call__(self, x: Tensor, xa: Optional[Tensor]=None, mask: Optional[Tensor]=None, len: Union[Variable,int]=None):
        if self.kv_caching == 'cross':
            if xa is not None:
                k, v = self.key(xa), self.value(xa)
                if not hasattr(self, 'cache_k'):
                    self.cache_k, self.cache_v = k, v
                else:
                    self.cache_k.assign(k).realize()
                    self.cache_v.assign(v).realize()
            else:
                k, v = self.cache_k, self.cache_v
        else:
            k, v = self.key(x), self.value(x)
            if self.kv_caching == 'self':
                if not hasattr(self, 'cache_k'):
                    self.cache_k = Tensor.zeros(x.shape[0], self.max_self_attn_cache_len, x.shape[2])
                    self.cache_v = Tensor.zeros(x.shape[0], self.max_self_attn_cache_len, x.shape[2])
                k = self.cache_k.shrink((None, (0, len), None)).cat(k, dim=1)
                v = self.cache_v.shrink((None, (0, len), None)).cat(v, dim=1)
                padding = self.max_self_attn_cache_len-len-x.shape[1]
                self.cache_k.assign(k.pad((None, (0, padding), None)).contiguous()).realize()
                self.cache_v.assign(v.pad((None, (0, padding), None)).contiguous()).realize()

        q = self.query(x)
        n_ctx = q.shape[1]
        assert(q.shape[-1] == k.shape[-1] == v.shape[-1])
        head_dim = q.shape[-1] // self.n_head
        q = q.reshape(*q.shape[:2], self.n_head, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(*k.shape[:2], self.n_head, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(*v.shape[:2], self.n_head, head_dim).permute(0, 2, 1, 3)
        attn = Tensor.scaled_dot_product_attention(q, k, v, mask[:n_ctx,:n_ctx] if mask is not None else None)
        wv = attn.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.out(wv)


class ResidualAttentionBlock:
    def __init__(self, n_state, n_head, is_decoder_block=False, max_self_attn_cache_len=None):
        self.attn = MultiHeadAttention(n_state, n_head, kv_caching='self' if is_decoder_block else None, max_self_attn_cache_len=max_self_attn_cache_len)
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head, kv_caching='cross') if is_decoder_block else None
        self.cross_attn_ln = nn.LayerNorm(n_state) if is_decoder_block else None

        self.mlp = [nn.Linear(n_state, n_state*4), Tensor.gelu, nn.Linear(n_state*4, n_state)]
        self.mlp_ln = nn.LayerNorm(n_state)

    def __call__(self, x, xa=None, mask=None, len: Union[Variable, int]=None):
        x = x + self.attn(self.attn_ln(x), mask=mask, len=len)
        if self.cross_attn: 
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)
        x = x + self.mlp_ln(x).sequential(self.mlp)
        return x.realize()


class AudioEncoder:
    def __init__(self, n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, **_):
        self.conv1 = nn.Conv1d(n_mels, n_audio_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_audio_state, n_audio_state, kernel_size=3, stride=2, padding=1)
        self.blocks = [ResidualAttentionBlock(n_audio_state, n_audio_head) for _ in range(n_audio_layer)]
        self.ln_post = nn.LayerNorm(n_audio_state)
        
        # Initialize positional embedding with proper values
        self.positional_embedding = Tensor(sinusoids(n_audio_ctx, n_audio_state))
        self.encode = TinyJit(self.__call__)

    def __call__(self, x):
        x = self.conv1(x).gelu()
        x = self.conv2(x).gelu()
        x = x.permute(0, 2, 1)
        x = x + self.positional_embedding[:x.shape[1]]
        x = x.sequential(self.blocks)
        x = self.ln_post(x)
        return x.realize()


class TextDecoder:
    def __init__(self, n_vocab, n_text_ctx, n_text_state, n_text_head, n_text_layer, **_):
        self.max_tokens_to_sample = n_text_ctx // 2
        self.max_self_attn_cache_len = n_text_ctx

        self.token_embedding = nn.Embedding(n_vocab, n_text_state)
        self.positional_embedding = Tensor.empty(n_text_ctx, n_text_state)
        self.blocks = [ResidualAttentionBlock(n_text_state, n_text_head, is_decoder_block=True, max_self_attn_cache_len=self.max_self_attn_cache_len) for _ in range(n_text_layer)]
        self.ln = nn.LayerNorm(n_text_state)
        self.mask = Tensor.full((n_text_ctx, n_text_ctx), -np.inf).triu(1).realize()
        self.getjitted = collections.defaultdict(lambda: TinyJit(self.forward))

    def __call__(self, x: Tensor, pos: int, encoded_audio: Tensor):
        pos = Variable("self_attn_cache_len", 1, self.max_self_attn_cache_len-1).bind(pos) if pos else 0
        return self.getjitted[x.shape](x, pos, encoded_audio)

    def forward(self, x: Tensor, pos: Union[Variable, Literal[0]], encoded_audio: Tensor):
        seqlen = x.shape[-1]
        x = self.token_embedding(x) + self.positional_embedding.shrink(((pos, pos+seqlen), None, None))
        for block in self.blocks: 
            x = block(x, xa=encoded_audio, mask=self.mask, len=pos)
        return self.output_tok(x)

    def output_tok(self, x):
        return (self.ln(x) @ self.token_embedding.weight.T).realize()


class Whisper:
    def __init__(self, dims, batch_size=1):
        if isinstance(dims, ModelDimensions):
            dims = {
                'n_mels': dims.n_mels,
                'n_audio_ctx': dims.n_audio_ctx,
                'n_audio_state': dims.n_audio_state,
                'n_audio_head': dims.n_audio_head,
                'n_audio_layer': dims.n_audio_layer,
                'n_vocab': dims.n_vocab,
                'n_text_ctx': dims.n_text_ctx,
                'n_text_state': dims.n_text_state,
                'n_text_head': dims.n_text_head,
                'n_text_layer': dims.n_text_layer,
            }
        
        self.encoder = AudioEncoder(**dims)
        self.decoder = TextDecoder(**dims)
        self.dims = ModelDimensions(**dims) if isinstance(dims, dict) else dims
        self.is_multilingual = dims.get("n_vocab", dims.n_vocab if hasattr(dims, 'n_vocab') else 0) >= 51865
        self.batch_size = batch_size

    def embed_audio(self, mel: Tensor):
        return self.encoder(mel)

    def logits(self, tokens: Tensor, audio_features: Tensor):
        return self.decoder(tokens, 0, audio_features)

    def __call__(self, mel: Tensor, tokens: Tensor) -> Tensor:
        return self.decoder(tokens, 0, self.encoder(mel))

    @property
    def device(self):
        return "tinygrad"  # Placeholder for compatibility

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def set_alignment_heads(self, dump: bytes):
        """Set alignment heads (not implemented in tinygrad version)"""
        import warnings
        warnings.warn("set_alignment_heads not implemented in tinygrad version")

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """Install KV cache hooks (not needed in tinygrad version due to different caching)"""
        import warnings
        warnings.warn("install_kv_cache_hooks not needed in tinygrad version")
        return {}, []

    # Add the function references for compatibility
    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function