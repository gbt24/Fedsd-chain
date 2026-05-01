# -*- coding: UTF-8 -*-
from blockchain.hash_utils import sha256_json, sha256_text


def merkle_leaf_hash(record):
    return sha256_json(record)


def _hash_pair(left_hash, right_hash):
    return sha256_text(left_hash + right_hash)


def build_merkle_tree(leaf_hashes):
    if not leaf_hashes:
        raise ValueError("leaf_hashes must not be empty")

    levels = [list(leaf_hashes)]
    current = list(leaf_hashes)
    while len(current) > 1:
        next_level = []
        for index in range(0, len(current), 2):
            left_hash = current[index]
            right_hash = current[index + 1] if index + 1 < len(current) else current[index]
            next_level.append(_hash_pair(left_hash, right_hash))
        levels.append(next_level)
        current = next_level
    return {"levels": levels, "root": levels[-1][0]}


def get_merkle_root(leaf_hashes):
    return build_merkle_tree(leaf_hashes)["root"]


def get_merkle_proof(leaf_hashes, index):
    tree = build_merkle_tree(leaf_hashes)
    proof = []
    current_index = index
    for level in tree["levels"][:-1]:
        if current_index % 2 == 0:
            sibling_index = current_index + 1 if current_index + 1 < len(level) else current_index
            position = "right"
        else:
            sibling_index = current_index - 1
            position = "left"
        proof.append({"position": position, "hash": level[sibling_index]})
        current_index //= 2
    return proof


def verify_merkle_proof(leaf_hash, proof, root):
    current_hash = leaf_hash
    for step in proof:
        if step["position"] == "left":
            current_hash = _hash_pair(step["hash"], current_hash)
        else:
            current_hash = _hash_pair(current_hash, step["hash"])
    return current_hash == root
