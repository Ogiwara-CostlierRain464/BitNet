import os
import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm

def align_w2_sparsity(ckpt_dir: str, output_path: str):
    input_path = Path(ckpt_dir) / "model_state_fp16.pt"
    print(f"Loading checkpoint from: {input_path}")

    # CPUメモリ上で安全にロード
    state_dict = torch.load(input_path, map_location="cpu", weights_only=True)

    target_sparsity_count = 4096

    # state_dictの中から "feed_forward.w2.weight" を含むキーだけを抽出
    target_keys = [k for k in state_dict.keys() if "feed_forward.w2.weight" in k]
    print(f"Found {len(target_keys)} 'w2' weight matrices to process.")

    # プログレスバーを表示しながらレイヤーごとに処理
    for key in tqdm(target_keys, desc="Adjusting Sparsity"):
        tensor = state_dict[key]

        # PyTorchの nn.Linear の weight は (out_features, in_features) = (2560, 6912) ですが、
        # どちらの次元が6912であっても対応できるように汎用的に判定します。
        if tensor.shape[0] == 6912:
            dim_to_process = 0
            num_vectors = tensor.shape[1]
        elif tensor.shape[1] == 6912:
            dim_to_process = 1
            num_vectors = tensor.shape[0]
        else:
            print(f"\nWarning: Neither dimension is 6912 for {key} (shape: {tensor.shape}). Skipping.")
            continue

        # 各列(または各行)の長さ6912のベクトルを取り出して処理
        for i in range(num_vectors):
            # 元のテンソルへの参照(view)として取得されるため、vecを更新するとtensor全体も更新されます
            if dim_to_process == 1:
                vec = tensor[i, :]
            else:
                vec = tensor[:, i]

            # 非ゼロ要素と0の要素のインデックスを抽出
            # nonzero() は shape [N, 1] を返すので、squeeze(-1) で1次元にしてリスト化
            nonzero_indices = torch.nonzero(vec).squeeze(-1).tolist()
            zero_indices = torch.nonzero(vec == 0).squeeze(-1).tolist()

            current_count = len(nonzero_indices)

            if current_count < target_sparsity_count:
                # 【不足している場合】
                # 0 の要素からランダムに不足分を選び、-1 と 1 にする
                diff = target_sparsity_count - current_count
                indices_to_fill = random.sample(zero_indices, diff)

                # -1と1を同数用意（奇数の場合は1が1つ多くなる）
                num_ones = diff // 2 + (diff % 2)
                num_minus_ones = diff // 2

                values = [1.0] * num_ones + [-1.0] * num_minus_ones
                random.shuffle(values)

                # 元のテンソルの型（fp16等）に合わせて代入
                values_tensor = torch.tensor(values, dtype=vec.dtype, device=vec.device)
                indices_tensor = torch.tensor(indices_to_fill, dtype=torch.long, device=vec.device)

                vec[indices_tensor] = values_tensor

            elif current_count > target_sparsity_count:
                # 【多すぎる場合】
                # 非ゼロ要素 からランダムに超過分を選び、0 にする
                diff = current_count - target_sparsity_count
                indices_to_zero = random.sample(nonzero_indices, diff)

                indices_tensor = torch.tensor(indices_to_zero, dtype=torch.long, device=vec.device)
                vec[indices_tensor] = 0.0

    # 保存先ディレクトリの作成と出力
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"\nSaving updated checkpoint to: {output_path}")
    torch.save(state_dict, output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adjust sparsity of w2 weights in Transformer feed-forward layers.")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="元の model_state_fp16.pt があるディレクトリ")
    parser.add_argument("--output", type=str, default="./checkpoints/aligned_randomly.pt", help="更新後のチェックポイントの保存先")
    args = parser.parse_args()

    align_w2_sparsity(args.ckpt_dir, args.output)