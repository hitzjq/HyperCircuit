#!/usr/bin/env python
import argparse
import os
import sys

import torch


def shard_tag(shard_index, num_shards):
    return f"shard_{shard_index:02d}_of_{num_shards:02d}"


def chunk_name(chunk_index, num_chunks):
    return f"chunk_{chunk_index:03d}_of_{num_chunks:03d}.pt"


def merge_one_shard(shards_root, shard_index, num_shards, num_chunks, force=False):
    tag = shard_tag(shard_index, num_shards)
    shard_root = os.path.join(shards_root, tag)
    chunk_root = os.path.join(shard_root, "vector_chunks")
    output_path = os.path.join(shard_root, "feat.pt")
    legacy_output_path = os.path.join(
        shard_root, f"cc_advanced_features_test_{tag}.pt"
    )

    if os.path.isfile(output_path) and not force:
        print(f"[skip] {tag}: feature already exists: {output_path}")
        return output_path

    if os.path.isfile(legacy_output_path) and not force:
        print(f"[skip] {tag}: legacy feature already exists: {legacy_output_path}")
        return legacy_output_path

    if not os.path.isdir(chunk_root):
        raise FileNotFoundError(f"{tag}: missing chunk directory: {chunk_root}")

    features = []
    query_mapping = []
    missing = []
    feature_dim = None
    source_n_graphs = None

    for chunk_index in range(num_chunks):
        path = os.path.join(chunk_root, chunk_name(chunk_index, num_chunks))
        if not os.path.isfile(path):
            missing.append(path)
            continue

        data = torch.load(path, map_location="cpu", weights_only=False)
        chunk_features = data["features"]
        if feature_dim is None:
            feature_dim = int(data.get("feature_dim", chunk_features.shape[1]))
        elif chunk_features.shape[1] != feature_dim:
            raise ValueError(
                f"{tag}: chunk {chunk_index} feature_dim mismatch: "
                f"{chunk_features.shape[1]} != {feature_dim}"
            )

        if data.get("chunk_index") not in (None, chunk_index):
            raise ValueError(
                f"{tag}: chunk metadata mismatch for {path}: "
                f"chunk_index={data.get('chunk_index')}"
            )
        if data.get("num_chunks") not in (None, num_chunks):
            raise ValueError(
                f"{tag}: chunk metadata mismatch for {path}: "
                f"num_chunks={data.get('num_chunks')}"
            )

        if source_n_graphs is None:
            source_n_graphs = data.get("source_n_graphs")

        features.append(chunk_features)
        query_mapping.extend(data["query_mapping"])

    if missing:
        preview = "\n".join(missing[:10])
        extra = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        raise FileNotFoundError(
            f"{tag}: missing {len(missing)}/{num_chunks} chunk files:\n"
            f"{preview}{extra}"
        )

    if not features:
        raise RuntimeError(f"{tag}: no chunk features found in {chunk_root}")

    merged_features = torch.cat(features, dim=0)
    if len(query_mapping) != merged_features.shape[0]:
        raise ValueError(
            f"{tag}: query mapping length mismatch: "
            f"{len(query_mapping)} != {merged_features.shape[0]}"
        )
    if source_n_graphs is not None and merged_features.shape[0] != source_n_graphs:
        raise ValueError(
            f"{tag}: merged rows do not match source graph count: "
            f"{merged_features.shape[0]} != {source_n_graphs}"
        )

    output_data = {
        "features": merged_features,
        "query_mapping": query_mapping,
        "feature_dim": merged_features.shape[1],
        "n_queries": len(query_mapping),
        "num_chunks": num_chunks,
        "source": "merge_vector_chunks",
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(output_data, output_path)
    print(f"[merged] {tag}: {merged_features.shape[0]} rows -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Merge graph_to_vector chunk outputs.")
    parser.add_argument("--run_name", default="prod_0421_1742")
    parser.add_argument("--num_shards", type=int, default=40)
    parser.add_argument("--num_chunks", type=int, default=64)
    parser.add_argument("--shard_index", type=int, default=-1,
                        help="Merge one shard only; -1 merges all shards.")
    parser.add_argument("--shards_root", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    shards_root = args.shards_root or os.path.join(
        "CodeCircuit_TRM_Arc1", "runs", args.run_name, "test_shards"
    )

    if args.shard_index >= 0:
        shard_indices = [args.shard_index]
    else:
        shard_indices = list(range(args.num_shards))

    print("==========================================================")
    print("  Merge vector chunks")
    print(f"  Run: {args.run_name}")
    print(f"  Shards root: {shards_root}")
    print(f"  Shards: {shard_indices}")
    print(f"  Chunks per shard: {args.num_chunks}")
    print("==========================================================")

    for shard_index in shard_indices:
        merge_one_shard(
            shards_root=shards_root,
            shard_index=shard_index,
            num_shards=args.num_shards,
            num_chunks=args.num_chunks,
            force=args.force,
        )

    print("All requested shard chunk merges completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
