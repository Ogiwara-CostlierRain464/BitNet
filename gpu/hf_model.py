# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Optional, Tuple, Union, List, Dict, Any

import torch
from torch import nn
from torch.nn import functional as F

from xformers.ops import RMSNorm, fmha, rope_padded
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
)

import ctypes
from pathlib import Path

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

_this_dir = Path(__file__).resolve().parent
lib_path = _this_dir / "bitnet_kernels" / "libbitnet.so"

# [FIX] カスタムカーネル (.so) は現状どこからも呼ばれていない。
# 存在しない環境で import ごと落ちるのを避けるため、遅延・寛容ロードにする。
bitnet_lib = None
try:
    if lib_path.exists():
        bitnet_lib = ctypes.CDLL(str(lib_path))
except OSError as e:  # pragma: no cover
    warnings.warn(f"libbitnet.so のロードに失敗しました（カーネル未使用のため続行します）: {e}")

from tokenizer import Tokenizer, ChatFormat, Message


class BitNetConfig(PretrainedConfig):
    model_type = "bitnet"

    def __init__(
        self,
        dim: int = 2560,
        n_layers: int = 30,
        n_heads: int = 20,
        n_kv_heads: int = 5,
        vocab_size: int = 128256,
        ffn_dim: int = 6912,
        norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        max_seq_len: int = 2048,
        use_cache: bool = True,  # [FIX] KVキャッシュ経路を修正したので既定で有効に
        **kwargs,
    ):
        # BitNet specific
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.ffn_dim = ffn_dim
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.use_cache = use_cache

        # Hugging Face compatibility
        self.hidden_size = dim
        self.num_hidden_layers = n_layers
        self.num_attention_heads = n_heads
        self.num_key_value_heads = n_kv_heads
        self.intermediate_size = ffn_dim
        self.rms_norm_eps = norm_eps

        # [FIX] tok_embeddings と output は別々の重み。
        # post_init() 内の tie_weights() が output.weight を潰さないよう明示的に無効化する。
        kwargs.setdefault("tie_word_embeddings", False)

        super().__init__(**kwargs)


LayerCache = Tuple[torch.Tensor, torch.Tensor]


class BitLinear(nn.Linear):
    @torch.compile
    def quant_input(self, input):
        s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (input * s).round().clamp(-128, 127) / s

    def forward(self, input):
        input = self.quant_input(input)
        # NOTE: 重みはチェックポイント側で量子化済みの想定なので、ここでは量子化しない。
        return F.linear(input, self.weight)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        norm_eps: float,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_local_heads = n_heads
        self.n_local_kv_heads = n_kv_heads

        Linear = BitLinear

        self.wqkv = Linear(
            dim,
            (self.n_local_heads + 2 * self.n_local_kv_heads) * head_dim,
            bias=False,
        )
        self.wo = Linear(
            self.n_local_heads * head_dim,
            dim,
            bias=False,
        )
        self.attn_sub_norm = RMSNorm(dim, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache,
        attn_bias: AttnBias,
    ) -> torch.Tensor:

        xqkv = self.wqkv(x)
        xq = xqkv[:, : (self.n_local_heads * self.head_dim)]
        xkv = xqkv[:, (self.n_local_heads * self.head_dim) :]
        xk, xv = xkv.chunk(2, 1)

        output_shape = xq.shape
        heads_per_group = self.n_local_heads // self.n_local_kv_heads
        xq = xq.view(
            1, xq.shape[0], self.n_local_kv_heads, heads_per_group, self.head_dim
        )
        xk = xk.view(1, xk.shape[0], self.n_local_kv_heads, 1, self.head_dim)
        xv = xv.view(1, xv.shape[0], self.n_local_kv_heads, 1, self.head_dim)
        cache_k, cache_v = cache

        # rope_padded は xq を回転して返しつつ、xk/xv を cache_k/cache_v に書き込む
        xq = rope_padded(
            xq=xq,
            xk=xk,
            xv=xv,
            cache_k=cache_k,
            cache_v=cache_v,
            attn_bias=attn_bias,
            theta=self.rope_theta,
        )

        output = fmha.memory_efficient_attention_forward(
            xq, cache_k, cache_v, attn_bias, op=fmha.flash.FwOp
        )

        output = output.reshape(output_shape)
        output = self.attn_sub_norm(output)
        output = self.wo(output)

        return output


@torch.compile
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) ** 2


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        norm_eps: float,
    ):
        super().__init__()

        Linear = BitLinear

        self.w13 = Linear(
            dim,
            2 * hidden_dim,
            bias=False,
        )
        self.w2 = Linear(
            hidden_dim,
            dim,
            bias=False,
        )
        self.ffn_sub_norm = RMSNorm(hidden_dim, norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x13 = self.w13(x)
        x1, x3 = x13.chunk(2, -1)
        inner = self.ffn_sub_norm(squared_relu(x1) * x3)
        output = self.w2(inner)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: BitNetConfig):
        super().__init__()

        assert config.dim % config.n_heads == 0
        head_dim = config.dim // config.n_heads
        n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_heads

        assert config.n_heads % n_kv_heads == 0

        # [FIX] use_kernel / use_sptmm / sparsity は一度も参照されていなかったので削除
        self.attention = Attention(
            dim=config.dim,
            head_dim=head_dim,
            n_heads=config.n_heads,
            n_kv_heads=n_kv_heads,
            rope_theta=config.rope_theta,
            norm_eps=config.norm_eps,
        )
        self.feed_forward = FeedForward(
            dim=config.dim,
            hidden_dim=config.ffn_dim,
            norm_eps=config.norm_eps,
        )
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache,
        attn_bias: AttnBias,
    ) -> torch.Tensor:
        # [FIX] _cached_cache / _cached_attn_bias によるモジュール属性経由の受け渡しを削除。
        # 呼び出し側が必ず明示的に渡すので、暗黙の状態を持たせない。
        if cache is None or attn_bias is None:
            raise ValueError("TransformerBlock.forward には cache と attn_bias が必須です")

        # 内部表現は packed な 2D (total_tokens, dim)。3D で来た場合だけ一時的に潰す。
        squeezed = False
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
            squeezed = True

        h = x + self.attention(self.attention_norm(x), cache, attn_bias)
        out = h + self.feed_forward(self.ffn_norm(h))

        if squeezed:
            out = out.unsqueeze(0)

        return out


class Transformer(nn.Module):
    def __init__(self, config: BitNetConfig):
        super().__init__()
        assert config.vocab_size > 0
        self.config = config

        self.tok_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.dim,
        )

        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    @property
    def embed_tokens(self):
        return self.tok_embeddings

    @embed_tokens.setter
    def embed_tokens(self, value):
        self.tok_embeddings = value

    @torch.no_grad()
    def forward_with_attn_bias(
        self,
        token_values: torch.Tensor,
        attn_bias: AttnBias,
        cache: List[LayerCache],
    ) -> torch.Tensor:
        """packed な 1D トークン列を受け取り、packed な 2D logits (total_tokens, vocab) を返す。"""
        if token_values.dim() != 1:
            token_values = token_values.reshape(-1)

        h = self.tok_embeddings(token_values)  # (total_tokens, dim)

        for i, layer in enumerate(self.layers):
            h = layer(h, cache=cache[i], attn_bias=attn_bias)

        logits = self.output(self.norm(h))
        return logits.float()

    @torch.no_grad()
    def forward(
        self,
        token_values: Optional[torch.Tensor] = None,
        token_lengths: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
        cache: Optional[List[LayerCache]] = None,
        kv_padding: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """
        token_values: packed 1D (total_tokens,) か、パディングなしの矩形 2D (B, T)
        token_lengths: (B,) 各シーケンスで「今回流す」トークン数
        start_pos:     (B,) 各シーケンスで「すでにキャッシュに入っている」トークン数
        cache:         make_cache() の戻り値。総長は batch_size * kv_padding であること
        kv_padding:    1シーケンスあたりのキャッシュ確保長。0 ならキャッシュ形状から推定
        """
        if token_values is None and "input_ids" in kwargs:
            token_values = kwargs.pop("input_ids")

        if token_values is None:
            raise ValueError("token_values or input_ids must be specified.")

        device = token_values.device

        rectangular_shape: Optional[Tuple[int, int]] = None
        if token_values.dim() == 2:
            # 矩形入力はパディングなしとして扱う
            bsz, seq_len = token_values.shape
            rectangular_shape = (bsz, seq_len)
            token_values = token_values.reshape(-1)
            token_lengths = torch.full((bsz,), seq_len, device=device, dtype=torch.int32)
        elif token_values.dim() == 1:
            if token_lengths is None:
                bsz = 1
                token_lengths = torch.tensor(
                    [token_values.shape[0]], device=device, dtype=torch.int32
                )
            else:
                bsz = int(token_lengths.numel())
        else:
            raise ValueError(f"token_values は 1D か 2D である必要があります (got {token_values.dim()}D)")

        token_lengths = token_lengths.to(device=device, dtype=torch.int32)

        if start_pos is None:
            start_pos = torch.zeros_like(token_lengths)
        start_pos = start_pos.to(device=device, dtype=torch.int32)

        if int(token_lengths.sum().item()) != token_values.shape[0]:
            raise ValueError(
                f"token_lengths の合計 ({int(token_lengths.sum().item())}) と "
                f"実際のトークン数 ({token_values.shape[0]}) が一致しません"
            )

        # [FIX] キャッシュ未指定時のみ確保。kv_padding も同時に決める。
        if cache is None:
            per_seq_len = max(int((start_pos + token_lengths).max().item()), 1)
            cache = make_cache(
                self.config,
                length=per_seq_len,
                batch_size=bsz,
                device=device,
                dtype=self.tok_embeddings.weight.dtype,
            )
            kv_padding = per_seq_len

        # [FIX] 元コードは max(kv_padding, cache全長) としていたため、
        # バッチ>1 で「1シーケンスあたりの長さ」と「バッファ全長」を取り違えていた。
        total_cache_len = cache[0][0].shape[1]
        if kv_padding <= 0:
            if total_cache_len % bsz != 0:
                raise ValueError(
                    f"キャッシュ全長 {total_cache_len} が batch_size {bsz} で割り切れません。"
                    " kv_padding を明示してください。"
                )
            kv_padding = total_cache_len // bsz

        if total_cache_len < bsz * kv_padding:
            raise ValueError(
                f"KVキャッシュが不足しています: 必要 {bsz * kv_padding} "
                f"(batch_size={bsz} x kv_padding={kv_padding}) / 実際 {total_cache_len}"
            )

        max_needed = int((start_pos + token_lengths).max().item())
        if max_needed > kv_padding:
            raise ValueError(
                f"系列長 {max_needed} が kv_padding {kv_padding} を超えました"
                " (max_seq_len を増やすか、履歴を切り詰めてください)"
            )

        attn_bias = AttnBias.from_seqlens(
            q_seqlen=token_lengths.tolist(),
            kv_seqlen=(start_pos + token_lengths).tolist(),
            kv_padding=kv_padding,
        )
        logits = self.forward_with_attn_bias(token_values, attn_bias, cache)

        if rectangular_shape is not None:
            logits = logits.view(rectangular_shape[0], rectangular_shape[1], -1)

        return logits


# generate() が渡してくるが、この実装では使わない引数（警告を出さない）
_IGNORED_FORWARD_KWARGS = frozenset(
    {
        "position_ids",
        "cache_position",
        "output_attentions",
        "output_hidden_states",
        "inputs_embeds",
        "num_items_in_batch",
    }
)


class BitnetForCausalLM(PreTrainedModel):
    config_class = BitNetConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _supports_cache_class = False  # 独自フォーマットのキャッシュを使う

    def __init__(self, config: BitNetConfig):
        super().__init__(config)
        self.model = Transformer(config)
        self.vocab_size = config.vocab_size
        # [FIX] post_init() が呼ばれていなかった。config.tie_word_embeddings=False なので
        # tie_weights() は no-op になり、output.weight が上書きされる事故は起きない。
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        return self.model.output

    def set_output_embeddings(self, new_embeddings):
        self.model.output = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @property
    def hf_device_map(self):
        if getattr(self, "_hf_device_map", None) is not None:
            return self._hf_device_map
        return {"": self.device}

    @hf_device_map.setter
    def hf_device_map(self, value):
        self._hf_device_map = value

    @property
    def seqlen(self) -> int:
        return getattr(self.config, "max_seq_len", 2048)

    def allocate_cache(
        self,
        batch_size: int = 1,
        max_seq_len: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[List[LayerCache], int]:
        """KVキャッシュを1回だけ確保するためのヘルパー。(cache, kv_padding) を返す。"""
        if max_seq_len is None:
            max_seq_len = self.config.max_seq_len
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.model.tok_embeddings.weight.dtype
        cache = make_cache(
            config=self.config,
            length=max_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        return cache, max_seq_len

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        token_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[LayerCache], torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # [FIX] 以下4つは元コードでは **kwargs に吸い込まれて黙って捨てられていた。
        token_lengths: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
        cache: Optional[List[LayerCache]] = None,
        kv_padding: int = 0,
        num_logits_to_keep: int = 0,
        **kwargs,
    ):
        """
        input_ids:      (B, T_new) 「今回処理する」トークンだけ（全履歴ではない）
        attention_mask: (B, T_new) input_ids と同じ長さ。長い場合は末尾 T_new を使う
        start_pos:      (B,) すでにキャッシュ済みのトークン数
        num_logits_to_keep: 0=全位置の logits を返す / 1=各系列の最終位置のみ (B,1,V)
        """
        # [FIX] 未知の引数を黙殺せず知らせる（同種の事故の再発防止）
        unexpected = set(kwargs) - _IGNORED_FORWARD_KWARGS
        if unexpected:
            warnings.warn(
                f"BitnetForCausalLM.forward: 未対応の引数を無視しました: {sorted(unexpected)}"
            )
        if "logits_to_keep" in kwargs:  # 新しい transformers での名称
            num_logits_to_keep = int(kwargs["logits_to_keep"])

        if input_ids is None and token_values is not None:
            input_ids = token_values
        elif input_ids is None and "input_0" in kwargs:
            input_ids = kwargs.pop("input_0")

        if input_ids is None:
            raise ValueError("You must specify input_ids or token_values")

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids は (B, T) である必要があります (got {tuple(input_ids.shape)})")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        elif attention_mask.shape[1] != seq_len:
            # generate() は全長の attention_mask を渡してくるので、今回流す分だけ切り出す
            if attention_mask.shape[1] < seq_len:
                raise ValueError(
                    f"attention_mask ({attention_mask.shape[1]}) が input_ids ({seq_len}) より短いです"
                )
            attention_mask = attention_mask[:, -seq_len:]

        # -------------------------------------------------------------
        # キャッシュの解決: 明示的な cache > past_key_values > 新規確保
        # -------------------------------------------------------------
        if not _is_valid_past(past_key_values):
            if past_key_values is not None:
                # HF が DynamicCache などを注入してきた場合は無視する
                warnings.warn(
                    "past_key_values の形式が想定外だったため無視しました "
                    "(期待する形式: (cache, start_pos) のタプル)"
                )
            past_key_values = None

        if cache is not None:
            cache_buffer = cache
            if start_pos is None:
                start_pos = torch.zeros(batch_size, dtype=torch.int32, device=device)
        elif past_key_values is not None:
            cache_buffer, start_pos = past_key_values
        else:
            # [FIX] dtype を bfloat16 決め打ちにせず、実際のパラメータに合わせる
            cache_buffer = make_cache(
                config=self.config,
                length=self.config.max_seq_len,
                batch_size=batch_size,
                device=device,
                dtype=self.model.tok_embeddings.weight.dtype,
            )
            kv_padding = self.config.max_seq_len
            start_pos = torch.zeros(batch_size, dtype=torch.int32, device=device)

        start_pos = start_pos.to(device=device, dtype=torch.int32)

        # [FIX] kv_padding は「1シーケンスあたり」の長さ。バッファ全長ではない。
        if kv_padding <= 0:
            total_cache_len = cache_buffer[0][0].shape[1]
            if total_cache_len % batch_size != 0:
                raise ValueError(
                    f"キャッシュ全長 {total_cache_len} が batch_size {batch_size} で割り切れません。"
                    " kv_padding を明示してください。"
                )
            kv_padding = total_cache_len // batch_size

        # -------------------------------------------------------------
        # 1次元へのパッキング（xformers要件）
        # -------------------------------------------------------------
        valid_mask = attention_mask.bool()
        packed_tokens = input_ids[valid_mask]
        packed_lengths = valid_mask.sum(dim=-1).to(torch.int32)

        if int(packed_lengths.min().item()) == 0:
            raise ValueError("attention_mask が全て 0 の行があります（空のシーケンスは扱えません）")

        next_lengths = start_pos + packed_lengths

        # -------------------------------------------------------------
        # 推論 (Prefill / Decode 共通)
        # -------------------------------------------------------------
        logits_1d = self.model(
            token_values=packed_tokens,
            token_lengths=packed_lengths,
            start_pos=start_pos,
            cache=cache_buffer,
            kv_padding=kv_padding,
        )  # (total_tokens, vocab)

        # -------------------------------------------------------------
        # アンパッキング（HF要件）
        # -------------------------------------------------------------
        if num_logits_to_keep == 1:
            # [FIX] 生成では最終位置しか使わないので (B, T, V) をまるごと確保しない。
            # vocab=128k / T=2048 だと fp32 で 1GB 超になるため。
            last_idx = (torch.cumsum(packed_lengths.long(), dim=0) - 1)
            logits = logits_1d[last_idx].unsqueeze(1)  # (B, 1, V)
        else:
            logits = torch.zeros(
                batch_size, seq_len, self.config.vocab_size,
                device=logits_1d.device, dtype=logits_1d.dtype,
            )
            # NOTE: パディング位置の logits は 0 のまま。左パディング前提で、
            # 生成時に参照されるのは各行の最終位置（必ず実トークン）だけ。
            logits[valid_mask] = logits_1d

        next_cache = (cache_buffer, next_lengths) if use_cache else None

        if not return_dict:
            return tuple(v for v in [logits, next_cache] if v is not None)

        return CausalLMOutputWithPast(
            loss=None,  # 推論専用
            logits=logits,
            past_key_values=next_cache,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        # [FIX] 元コードは input_ids を全長のまま渡していたため、KVキャッシュ使用時に
        # kv_seqlen = start_pos + token_lengths が二重カウントされて壊れていた。
        if _is_valid_past(past_key_values):
            _, start_pos = past_key_values
            past_length = int(start_pos.max().item())
            if past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # past_length >= input_ids.shape[1] の場合は未処理分のみが渡っていると見なす
        else:
            past_key_values = None

        if attention_mask is not None:
            attention_mask = attention_mask[:, -input_ids.shape[1] :]

        return {
            "input_ids": input_ids.contiguous(),
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "num_logits_to_keep": 1,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # [FIX] 元コードは past_key_values をそのまま返していたため、beam search が
        # 別ビームの文脈を混ぜた出力を黙って返していた。未対応なら明示的に落とす。
        raise NotImplementedError(
            "このキャッシュ形式は beam search に未対応です。num_beams=1 を使ってください。"
        )


def _is_valid_past(past_key_values: Any) -> bool:
    """past_key_values が (cache, start_pos) 形式かどうか。"""
    return (
        isinstance(past_key_values, (tuple, list))
        and len(past_key_values) == 2
        and isinstance(past_key_values[0], (list, tuple))
        and len(past_key_values[0]) > 0
        and isinstance(past_key_values[1], torch.Tensor)
    )


def make_cache(
    config: BitNetConfig,
    length: int,
    batch_size: int = 1,
    device: Optional[Union[str, torch.device]] = None,
    n_layers: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> List[LayerCache]:
    """
    length: 1シーケンスあたりのキャッシュ長 (= kv_padding)
    batch_size: バッチサイズ。バッファ全長は batch_size * length になる。
    """
    head_dim = config.dim // config.n_heads
    n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_heads
    n_local_kv_heads = n_kv_heads

    if n_layers is None:
        n_layers = config.n_layers

    # [FIX] BlockDiagonalCausalWithOffsetPaddedKeysMask は各バッチ要素が
    # kv_padding 個ぶんの領域を持つ前提。元コードは length しか確保しておらず、
    # batch_size > 1 で範囲外アクセスになっていた。
    shape = (1, batch_size * length, n_local_kv_heads, 1, head_dim)
    heads_per_group = config.n_heads // n_kv_heads
    expansion = (-1, -1, -1, heads_per_group, -1)
    return [
        (
            torch.zeros(shape, device=device, dtype=dtype).expand(expansion),
            torch.zeros(shape, device=device, dtype=dtype).expand(expansion),
        )
        for _ in range(n_layers)
    ]


def cache_prefix(cache: List[LayerCache], length: int) -> List[LayerCache]:
    if len(cache) > 0:
        assert cache[0][0].shape[1] >= length

    return [(ck[:, :length], cv[:, :length]) for ck, cv in cache]


def load_bitnet_from_fp16(
    checkpoint_path: str = "./checkpoints/model_state_fp16.pt",
    config: Optional[BitNetConfig] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> BitnetForCausalLM:
    if config is None:
        config = BitNetConfig()

    model = BitnetForCausalLM(config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # キー名（プレフィックス）の不一致を自動変換
    prefix = "model."
    new_state_dict = {}
    unmapped = []
    model_state_keys = set(model.state_dict().keys())

    for key, value in state_dict.items():
        if key in model_state_keys:
            new_state_dict[key] = value
        elif prefix + key in model_state_keys:
            new_state_dict[prefix + key] = value
        elif key.startswith(prefix) and key[len(prefix) :] in model_state_keys:
            new_state_dict[key[len(prefix) :]] = value
        else:
            # [FIX] 元コードはここで未知キーをそのまま積んでいたため、
            # strict=True が必ず例外を投げ、下の警告 print に到達できなかった。
            unmapped.append(key)

    if unmapped:
        print(f"[Warning] 対応先が見つからないキーを無視しました ({len(unmapped)}件): {unmapped[:10]}")

    # [FIX] strict=True は不一致時に例外を投げるので missing/unexpected を受け取れない。
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    if unexpected_keys:
        print(f"[Warning] Unexpected keys during loading: {unexpected_keys[:10]}")
    if missing_keys:
        raise RuntimeError(
            f"重みが不足しています ({len(missing_keys)}件): {missing_keys[:10]}\n"
            "チェックポイントのキー名とモデル構造が一致しているか確認してください。"
        )

    model = model.to(device=device, dtype=dtype)
    model.eval()

    return model


def encode_dialog(chat_formatter: ChatFormat, dialog: List[Message]) -> List[int]:
    """ChatFormat の API 差異を吸収してプロンプトをトークン化する。"""
    try:
        return chat_formatter.encode_dialog_prompt(dialog, completion=True)
    except TypeError:
        return chat_formatter.encode_dialog_prompt(dialog)


# ==========================================
# トークンサンプリング用関数
# ==========================================
def sample_next_token(logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits, dim=-1).item())

    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs = probs.clone()
    probs[indices_to_remove] = 0.0
    probs = probs / probs.sum().clamp(min=1e-9)  # [FIX] 0除算ガード

    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


# ==========================================
# model.generate を使わない自前生成ループ
# ==========================================
@torch.inference_mode()
def generate_custom(
    model: BitnetForCausalLM,
    prompt_tokens: List[int],
    stop_tokens: set,
    max_new_tokens: int = 512,
    max_seq_len: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
    cache: Optional[List[LayerCache]] = None,
) -> List[int]:
    prompt_len = len(prompt_tokens)
    if prompt_len == 0:
        return []
    if prompt_len >= max_seq_len:
        raise ValueError(f"プロンプトが長すぎます ({prompt_len} >= max_seq_len {max_seq_len})")

    # 1. KVキャッシュ領域の事前割り当て（呼び出し側で使い回せるよう引数でも受け取る）
    if cache is None:
        cache = make_cache(
            config=model.config,
            length=max_seq_len,
            batch_size=1,
            device=device,
            dtype=model.model.tok_embeddings.weight.dtype,
        )

    # --------------------------------------------------
    # 2. Prefill フェーズ
    # --------------------------------------------------
    tokens_tensor = torch.tensor([prompt_tokens], device=device, dtype=torch.long)
    start_pos = torch.tensor([0], device=device, dtype=torch.int32)

    # [FIX] 元コードはここで cache / start_pos / kv_padding を渡していたつもりが、
    # forward のシグネチャに存在せず **kwargs に落ちて捨てられていた。
    # その結果 decode 側は毎回まっさらなキャッシュ・start_pos=0 で動き、
    # 文脈を一切見ない（＝2トークン目以降が破綻する）状態だった。
    outputs = model(
        input_ids=tokens_tensor,
        start_pos=start_pos,
        cache=cache,
        kv_padding=max_seq_len,
        num_logits_to_keep=1,
        use_cache=True,
        return_dict=True,
    )
    next_token = sample_next_token(
        outputs.logits[0, -1, :], temperature=temperature, top_p=top_p
    )

    generated_tokens: List[int] = []
    cur_pos = prompt_len

    # --------------------------------------------------
    # 3. Decode フェーズ
    # --------------------------------------------------
    for _ in range(max_new_tokens):
        if next_token in stop_tokens or cur_pos >= max_seq_len - 1:
            break

        generated_tokens.append(next_token)

        next_input = torch.tensor([[next_token]], device=device, dtype=torch.long)
        start_pos = torch.tensor([cur_pos], device=device, dtype=torch.int32)

        outputs = model(
            input_ids=next_input,
            start_pos=start_pos,
            cache=cache,
            kv_padding=max_seq_len,
            num_logits_to_keep=1,
            use_cache=True,
            return_dict=True,
        )
        next_token = sample_next_token(
            outputs.logits[0, -1, :], temperature=temperature, top_p=top_p
        )
        cur_pos += 1

    return generated_tokens


def chat(
    model: BitnetForCausalLM,
    tokenizer: Tokenizer,
    chat_formatter: ChatFormat,
    device: str = "cuda",
    max_new_tokens: int = 512,
):
    """model.generate() を使う版（参考実装）。"""
    dialog: List[Message] = [
        {"role": "system", "content": "You are a helpful and polite AI assistant."}
    ]

    print("\nモデルの準備が完了しました。対話を開始します ('exit' で終了)\n" + "-" * 50)

    while True:
        try:
            user_input = input("\nUser > ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("対話を終了します。")
                break
            if not user_input.strip():
                continue

            dialog.append({"role": "user", "content": user_input})

            prompt_tokens = encode_dialog(chat_formatter, dialog)
            input_ids = torch.tensor([prompt_tokens], device=device, dtype=torch.long)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_beams=1,  # beam search は未対応
                    pad_token_id=tokenizer.pad_id,
                    eos_token_id=list(tokenizer.stop_tokens),
                )

            new_tokens = output_ids[0][len(prompt_tokens) :].tolist()

            cleaned_tokens = []
            for token in new_tokens:
                if token in tokenizer.stop_tokens:
                    break
                cleaned_tokens.append(token)

            # [FIX] ChatFormat ではなく Tokenizer が decode を持つ
            response_text = tokenizer.decode(cleaned_tokens).strip()
            print(f"Assistant > {response_text}")

            dialog.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            print("\n対話を終了します。")
            break


# ==========================================
# 対話実行メインループ
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. モデル構成とロード
    config = BitNetConfig(
        dim=2560,
        n_layers=30,
        n_heads=20,
        n_kv_heads=5,
        vocab_size=128256,
        ffn_dim=6912,
        max_seq_len=2048,
    )

    model = load_bitnet_from_fp16(
        checkpoint_path="./checkpoints/model_state_fp16.pt",
        config=config,
        device=device,
        dtype=torch.bfloat16,
    )

    # 2. Tokenizer & ChatFormat 初期化
    tokenizer_path = "./tokenizer.model"
    tokenizer = Tokenizer(model_path=tokenizer_path)
    chat_formatter = ChatFormat(tokenizer=tokenizer)

    # 3. KVキャッシュは一度だけ確保して使い回す
    # [FIX] 元コードは forward のたびに max_seq_len 分（bf16 で約160MB）を
    # 新規確保して捨てていた。
    cache, kv_padding = model.allocate_cache(batch_size=1, max_seq_len=config.max_seq_len)

    dialog: List[Message] = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

    print("\n[Low-Level Engine] モデルの準備が完了しました ('exit' で終了)\n" + "-" * 50)

    while True:
        try:
            user_input = input("\nUser > ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("終了します。")
                break
            if not user_input.strip():
                continue

            dialog.append({"role": "user", "content": user_input})

            # [FIX] チャットテンプレートと会話履歴が無効化されていた
            # （dialog を積んでいるのに tokenizer.encode(user_input) を使っていた）。
            prompt_tokens = encode_dialog(chat_formatter, dialog)

            generated_tokens = generate_custom(
                model=model,
                prompt_tokens=prompt_tokens,
                stop_tokens=tokenizer.stop_tokens,
                max_new_tokens=512,
                max_seq_len=kv_padding,
                temperature=0.7,
                top_p=0.9,
                device=device,
                cache=cache,
            )

            response_text = tokenizer.decode(generated_tokens).strip()
            print(f"Assistant > {response_text}")

            dialog.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            print("\n終了します。")
            break
        except ValueError as e:
            # 履歴が max_seq_len を超えた場合など
            print(f"[Error] {e}")
            dialog.pop()


if __name__ == "__main__":
    main()
