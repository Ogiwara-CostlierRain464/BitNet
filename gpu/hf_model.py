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

try:
    from transformers.modeling_outputs import CausalLMOutputWithPast
except ImportError:
    @dataclass
    class CausalLMOutputWithPast:
        logits: torch.Tensor = None
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
        hidden_states: Optional[Tuple[torch.Tensor]] = None
        attentions: Optional[Tuple[torch.Tensor]] = None

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
    _tied_weights_keys = ["lm_head.weight"]

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
        self.lm_head = self.model.output

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
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
    head_dim = args.dim // args.n_heads
    n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads

    if n_layers is None:
        n_layers = args.n_layers

    shape = (1, length, n_kv_heads, 1, head_dim)
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
        new_state_dict, strict=False
    )

    if missing_keys:
        print(f"[Warning] Missing keys during loading: {missing_keys}")
    if unexpected_keys:
        print(f"[Warning] Unexpected keys during loading: {unexpected_keys}")

    # 6. デバイス転送 & 精度変換 & 推論モード変更
    model = model.to(device=device, dtype=dtype)
    model.eval()

    return model


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Hugging Face 標準の generate を利用
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 生成された新規トークン部分のみをデコード
    new_tokens = output_ids[0][input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


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


if __name__ == "__main__":
    # モデル設定情報
    args = ModelArgs(
        dim=2560,
        n_layers=30,
        n_heads=20,
        n_kv_heads=5,
        vocab_size=128256,
        ffn_dim=6912,
        use_kernel=False,  # FP16からのロード時はカーネル指定をFalseまたは元設定に合わせます
    )

    # ロードの実行
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_bitnet_from_fp16(
        checkpoint_path="./checkpoints/model_state_fp16.pt",
        args=args,
        device=device,
        dtype=torch.bfloat16,
    )

    print("モデルのロードが完了しました。")

    tokenizer = Tokenizer("./tokenizer.model")
    chat_formatter = ChatFormat(tokenizer=tokenizer)
    chat(model, tokenizer, chat_formatter, device=device)


    # テスト推論（動作確認）
#     input_ids = torch.tensor([[1, 1504, 230]], device=device)
#     with torch.no_grad():
#         output = model(input_ids=input_ids, kv_padding = input_ids.shape[1])
#         print(f"Logits shape: {output.logits.shape}")