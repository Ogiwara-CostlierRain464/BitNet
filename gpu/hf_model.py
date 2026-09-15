# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

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

bitnet_lib = ctypes.CDLL(str(lib_path))

from tokenizer import Tokenizer, ChatFormat


@dataclass
class ModelArgs:
    dim: int = 2560
    n_layers: int = 30
    n_heads: int = 20
    n_kv_heads: int = 5
    vocab_size: int = 128256
    ffn_dim: int = 6912
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_kernel: bool = False
    use_sptmm: bool = False
    sparsity: int = 40 # 40 or 60 or 80


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
        use_kernel: bool = False,
        use_sptmm: bool = False,
        sparsity: int = 40,
        max_seq_len: int = 2048,
        use_cache: bool = False,
        **kwargs,
    ):
        self.dim = dim
        self.hidden_size = dim
        self.n_layers = n_layers
        self.num_hidden_layers = n_layers
        self.n_heads = n_heads
        self.num_attention_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.num_key_value_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.ffn_dim = ffn_dim
        self.intermediate_size = ffn_dim
        self.norm_eps = norm_eps
        self.rms_norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.use_kernel = use_kernel
        self.use_sptmm = use_sptmm
        self.sparsity = sparsity
        self.max_seq_len = max_seq_len
        self.use_cache = use_cache
        super().__init__(**kwargs)


LayerCache = Tuple[torch.Tensor, torch.Tensor]


class BitLinear(nn.Linear):
    @torch.compile
    def quant_input(self, input):
        s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (input * s).round().clamp(-128, 127) / s

    def forward(self, input):
        input = self.quant_input(input)
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
        use_kernel: bool,
        use_sptmm: bool,
        sparsity: int
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
        use_kernel: bool,
        use_sptmm: bool,
        sparsity: int
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
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.dim % args.n_heads == 0
        head_dim = args.dim // args.n_heads
        n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads

        assert args.n_heads % n_kv_heads == 0

        self.attention = Attention(
            dim=args.dim,
            head_dim=head_dim,
            n_heads=args.n_heads,
            n_kv_heads=n_kv_heads,
            rope_theta=args.rope_theta,
            norm_eps=args.norm_eps,
            use_kernel=args.use_kernel,
            use_sptmm=args.use_sptmm,
            sparsity=args.sparsity
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.ffn_dim,
            norm_eps=args.norm_eps,
            use_kernel=args.use_kernel,
            use_sptmm=args.use_sptmm,
            sparsity=args.sparsity
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[LayerCache] = None,
        attn_bias: Optional[AttnBias] = None,
        **kwargs
    ) -> torch.Tensor:
        if cache is None:
            cache = getattr(self, "_cached_cache", None)

        if attn_bias is None:
            attn_bias = getattr(self, "_cached_attn_bias", None)

        is_unsqueezed = False
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
            is_unsqueezed = True

        h = x + self.attention.forward(
            self.attention_norm(x),
            cache,
            attn_bias,
        )
        out = h + self.feed_forward(self.ffn_norm(h))

        if is_unsqueezed:
            out = out.unsqueeze(0)

        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size > 0
        self.args = args

        self.tok_embeddings = nn.Embedding(
            num_embeddings=args.vocab_size,
            embedding_dim=args.dim,
        )

        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

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

        for i, layer in enumerate(self.layers):
            target_layer = layer.module if hasattr(layer, "module") else layer
            target_layer._cached_cache = cache[i]
            target_layer._cached_attn_bias = attn_bias

        orig_shape = token_values.shape
        if token_values.dim() == 2 and token_values.size(0) > 1:
            token_values_flat = token_values.view(-1)
        else:
            token_values_flat = token_values

        h = self.tok_embeddings(token_values_flat)

        for i, layer in enumerate(self.layers):
            h = layer(h, cache=cache[i], attn_bias=attn_bias)

        logits = self.output(self.norm(h))
        logits = logits.float()

        if token_values.dim() == 2 and token_values.size(0) > 1:
            logits = logits.view(orig_shape[0], orig_shape[1], -1)

        return logits

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
        if token_values is None and "input_ids" in kwargs:
            token_values = kwargs["input_ids"]

        if token_values is None:
            raise ValueError("token_values or input_ids must be specified.")

        device = token_values.device
        if token_values.dim() == 2:
            bsz, seq_len = token_values.shape
        elif token_values.dim() == 1:
            bsz, seq_len = 1, token_values.shape[0]
        else:
            bsz, seq_len = 1, token_values.numel()

        if token_lengths is None:
            token_lengths = torch.tensor([seq_len] * bsz, device=device, dtype=torch.int32)

        if start_pos is None:
            start_pos = torch.zeros_like(token_lengths)

        if cache is None:
            total_len = int((start_pos + token_lengths).max().item())
            cache = make_cache(
                self.args,
                length=max(total_len, 1),
                device=device,
                dtype=self.tok_embeddings.weight.dtype,
            )

        attn_bias = AttnBias.from_seqlens(
            q_seqlen=token_lengths.tolist(),
            kv_seqlen=(start_pos + token_lengths).tolist(),
            kv_padding=kv_padding,
        )
        return self.forward_with_attn_bias(token_values, attn_bias, cache)


class BitnetForCausalLM(nn.Module):

    def __init__(self, args: Union[ModelArgs, any]):
        super().__init__()
        if not isinstance(args, ModelArgs):
            if hasattr(args, "to_dict"):
                cfg = args.to_dict()
            elif isinstance(args, dict):
                cfg = args
            else:
                cfg = getattr(args, "__dict__", {})

            model_args = ModelArgs(
                dim=cfg.get("dim", cfg.get("hidden_size", 2560)),
                n_layers=cfg.get("n_layers", cfg.get("num_hidden_layers", 30)),
                n_heads=cfg.get("n_heads", cfg.get("num_attention_heads", 20)),
                n_kv_heads=cfg.get("n_kv_heads", cfg.get("num_key_value_heads", 5)),
                vocab_size=cfg.get("vocab_size", 128256),
                ffn_dim=cfg.get("ffn_dim", cfg.get("intermediate_size", 6912)),
                norm_eps=cfg.get("norm_eps", cfg.get("rms_norm_eps", 1e-5)),
                rope_theta=cfg.get("rope_theta", 500000.0),
                use_kernel=cfg.get("use_kernel", False),
                use_sptmm=cfg.get("use_sptmm", False),
                sparsity=cfg.get("sparsity", 40),
            )
            args = model_args

        self.args = args
        self.config = args
        self.model = Transformer(args)
        self.vocab_size = args.vocab_size

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

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[LayerCache]] = None,
        return_dict: Optional[bool] = None,
        token_values: Optional[torch.Tensor] = None,
        token_lengths: Optional[torch.Tensor] = None,
        start_pos: Optional[torch.Tensor] = None,
        cache: Optional[List[LayerCache]] = None,
        kv_padding: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if input_ids is None and token_values is not None:
            input_ids = token_values
        elif input_ids is None and "input_0" in kwargs:
            input_ids = kwargs.pop("input_0")

        if input_ids is None:
            raise ValueError("You must specify input_ids or token_values")

        return_dict = (
            return_dict
            if return_dict is not None
            else getattr(self.args, "use_return_dict", True)
        )

        if cache is None and past_key_values is not None:
            cache = past_key_values


        logits = self.model(
            token_values=input_ids,
            token_lengths=token_lengths,
            start_pos=start_pos,
            cache=cache,
            kv_padding=kv_padding,
        )

        if not return_dict:
            return (logits,)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=cache,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, cache_position=None, **kwargs
    ):
        model_inputs = {"input_ids": input_ids.contiguous()}
        model_inputs.update(
            {
                "position_ids": kwargs.get("position_ids", None),
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return past_key_values


def make_cache(
    args: ModelArgs,
    length: int,
    device: Optional[Union[str, torch.device]] = None,
    n_layers: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> List[LayerCache]:
    """
    Allocate a cache to be used with the Transformer module.

    Args:
        args (ModelArgs): the model configuration.
        length (int): per layer cache size.
            It is usually budgeted as ``max_batch * max_seq``
        device (torch.device, optional): the device on which
            the cache should be allocated.
        n_layers (int, optional): the number of layers to
            allocate a cache for (defaults to the model
            settings).
        dtype (torch.dtype, optional): the dtype to use for
            cache entries (defaults to the default dtype).

    Returns:
        The cache object to pass to ``Tranformer.forward``.
    """

    head_dim = args.dim // args.n_heads
    n_kv_heads = args.n_kv_heads
    if n_kv_heads is None:
        n_kv_heads = args.n_heads
    n_local_kv_heads = n_kv_heads

    if n_layers is None:
        n_layers = args.n_layers

    shape = (1, length, n_local_kv_heads, 1, head_dim)
    heads_per_group = args.n_heads // n_kv_heads
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
    args: ModelArgs = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> BitnetForCausalLM:
    # 1. ModelArgsの準備（未指定の場合はデフォルトパラメータを使用）
    if args is None:
        args = ModelArgs()

    # 2. モデルのインスタンス化
    model = BitnetForCausalLM(args)

    # 3. チェックポイントの読み込み (CPU上に展開)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # チェックポイントが dict 構造（'model' や 'state_dict' キー）でラップされている場合の抽出
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 4. キー名（プレフィックス）の不一致を自動変換
    # Transformer単体保存 (例: tok_embeddings.weight) ↔ BitnetForCausalLM (例: model.tok_embeddings.weight)
    new_state_dict = {}
    model_state_keys = set(model.state_dict().keys())

    for key, value in state_dict.items():
        if key in model_state_keys:
            new_state_dict[key] = value
        elif f"model.{key}" in model_state_keys:
            new_state_dict[f"model.{key}"] = value
        elif key.startswith("model.") and key[6:] in model_state_keys:
            new_state_dict[key[6:]] = value
        else:
            new_state_dict[key] = value

    # 5. 重みのロード
    missing_keys, unexpected_keys = model.load_state_dict(
        new_state_dict, strict=True
    )

    if missing_keys:
        print(f"[Warning] Missing keys during loading: {missing_keys}")
    if unexpected_keys:
        print(f"[Warning] Unexpected keys during loading: {unexpected_keys}")

    # 6. デバイス転送 & 精度変換 & 推論モード変更
    model = model.to(device=device, dtype=dtype)
    model.eval()

    return model


# def generate_response(
#     model,
#     tokenizer,
#     prompt: str,
#     max_new_tokens: int = 256,
#     device: str = "cuda",
# ) -> str:
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
#
#     # Hugging Face 標準の generate を利用
#     with torch.no_grad():
#         output_ids = model.generate(
#             input_ids=input_ids,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9,
#             pad_token_id=tokenizer.eos_token_id,
#         )
#
#     # 生成された新規トークン部分のみをデコード
#     new_tokens = output_ids[0][input_ids.shape[1] :]
#     return tokenizer.decode(new_tokens, skip_special_tokens=True)


def chat(
    model: BitnetForCausalLM,
    tokenizer: Tokenizer,
    chat_formatter: ChatFormat,
    device: str = "cuda",
    max_new_tokens: int = 512,
):
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

            # 会話履歴にユーザーの発言を追加
            dialog.append({"role": "user", "content": user_input})

            # プロンプトのトークン化 (アシスタントの返答待ち状態を生成)
            prompt_tokens = chat_formatter.encode_dialog_prompt(dialog, completion=True)
            input_ids = torch.tensor([prompt_tokens], device=device, dtype=torch.long)

            # 生成実行
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_id,
                    eos_token_id=list(tokenizer.stop_tokens),
                )

            # 新規生成されたトークンのみ取り出し
            new_tokens = output_ids[0][len(prompt_tokens) :].tolist()

            # stop_tokens (<|end_of_text|> や <|eot_id|>) でカット
            cleaned_tokens = []
            for token in new_tokens:
                if token in tokenizer.stop_tokens:
                    break
                cleaned_tokens.append(token)

            response_text = chat_formatter.decode(cleaned_tokens).strip()
            print(f"Assistant > {response_text}")

            # 会話履歴にアシスタントの返答を追加
            dialog.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            print("\n対話を終了します。")
            break


# ==========================================
# トークンサンプリング用関数
# ==========================================
def sample_next_token(logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9) -> int:
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1).item()

    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    probs[indices_to_remove] = 0.0
    probs = probs / probs.sum()

    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()


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
) -> List[int]:
    prompt_len = len(prompt_tokens)

    # 1. KVキャッシュ領域の事前割り当て
    cache = make_cache(
        args=model.model.args,
        length=max_seq_len,
        device=device,
        dtype=model.model.tok_embeddings.weight.dtype,
    )

    # --------------------------------------------------
    # 2. Prefill フェーズ (プロンプトを一括処理してKVキャッシュを初期化)
    # --------------------------------------------------
    tokens_tensor = torch.tensor([prompt_tokens], device=device, dtype=torch.long)
    token_lengths = torch.tensor([prompt_len], device=device, dtype=torch.int32)
    start_pos = torch.tensor([0], device=device, dtype=torch.int32)

    outputs = model(
        token_values=tokens_tensor,
        token_lengths=token_lengths,
        start_pos=start_pos,
        cache=cache,
        kv_padding=max_seq_len,
        return_dict=True,
    )
    logits = outputs.logits

    # 最後のトークンの logits から最初の生成トークンをサンプリング
    last_logits = logits[0, -1, :]
    next_token = sample_next_token(last_logits, temperature=temperature, top_p=top_p)

    generated_tokens = []
    cur_pos = prompt_len

    # --------------------------------------------------
    # 3. Decode フェーズ (1トークンずつKVキャッシュに追記して自動生成)
    # --------------------------------------------------
    for _ in range(max_new_tokens):
        if next_token in stop_tokens or cur_pos >= max_seq_len - 1:
            break

        generated_tokens.append(next_token)

        # 1トークンのみ入力
        next_input = torch.tensor([[next_token]], device=device, dtype=torch.long)
        token_lengths = torch.tensor([1], device=device, dtype=torch.int32)
        start_pos = torch.tensor([cur_pos], device=device, dtype=torch.int32)

        outputs = model(
            token_values=next_input,
            token_lengths=token_lengths,
            start_pos=start_pos,
            cache=cache,
            kv_padding=max_seq_len,
            return_dict=True
        )
        logits = outputs.logits

        last_logits = logits[0, -1, :]
        next_token = sample_next_token(last_logits, temperature=temperature, top_p=top_p)
        cur_pos += 1

    return generated_tokens


# ==========================================
# 対話実行メインループ
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. モデル構成とロード
    args = ModelArgs(
        dim=2560,
        n_layers=30,
        n_heads=20,
        n_kv_heads=5,
        vocab_size=128256,
        ffn_dim=6912,
        use_kernel=False,
    )

    model = load_bitnet_from_fp16(
        checkpoint_path="./checkpoints/model_state_fp16.pt",
        args=args,
        device=device,
        dtype=torch.bfloat16
    )


    # 2. Tokenizer & ChatFormat 初期化
    tokenizer_path = "./tokenizer.model"
    tokenizer = Tokenizer(model_path=tokenizer_path)
    chat_formatter = ChatFormat(tokenizer=tokenizer)

    dialog: List[Message] = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

    print("\n[Low-Level Engine] モデルの準備が完了しました ('exit' で終了)\n" + "-" * 50)

    # 3. インタラクティブ対話ループ
    while True:
        try:
            user_input = input("\nUser > ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("終了します。")
                break
            if not user_input.strip():
                continue

            dialog.append({"role": "user", "content": user_input})
            #prompt_tokens = chat_formatter.encode_dialog_prompt(dialog, completion=True)
            prompt_tokens = tokenizer.encode(user_input, bos=False, eos=False)

            # 自前生成エンジンの呼び出し
            generated_tokens = generate_custom(
                model=model,
                prompt_tokens=prompt_tokens,
                stop_tokens=tokenizer.stop_tokens,
                max_new_tokens=512,
                max_seq_len=2048,
                temperature=0.7,
                top_p=0.9,
                device=device,
            )

            #response_text = chat_formatter.decode(generated_tokens).strip()
            response_text = tokenizer.decode(generated_tokens).strip()
            print(f"Assistant > {response_text}")

            dialog.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            print("\n終了します。")
            break


if __name__ == "__main__":
    main()