# -*- coding: UTF-8 -*-
import os
import json

from blockchain.evm_anchor import EVMAnchorClient
from blockchain.mock_anchor import MockAnchorClient


def create_anchor_client_from_args(args):
    if not getattr(args, "enable_anchor", False):
        return None

    mode = getattr(args, "anchor_mode", "mock")
    if mode == "mock":
        anchor_path = getattr(args, "anchor_path", None)
        if not anchor_path:
            save_dir = getattr(args, "save_dir", ".")
            anchor_path = os.path.join(save_dir, "blockchain", "anchors.jsonl")
        return MockAnchorClient(anchor_path)

    if mode == "evm":
        rpc_url = getattr(args, "anchor_rpc", None)
        contract_address = getattr(args, "anchor_contract", None)
        if not rpc_url or not contract_address:
            raise ValueError("EVM anchoring requires --anchor_rpc and --anchor_contract")
        abi = None
        abi_path = getattr(args, "anchor_abi", None)
        if abi_path:
            with open(abi_path, "r") as f:
                abi_payload = json.load(f)
            abi = abi_payload.get("abi", abi_payload)
        receipts_dir = getattr(args, "anchor_receipts_dir", None)
        if not receipts_dir:
            save_dir = getattr(args, "save_dir", ".")
            receipts_dir = os.path.join(save_dir, "blockchain", "anchor_receipts")
        return EVMAnchorClient(
            rpc_url=rpc_url,
            contract_address=contract_address,
            abi=abi,
            receipts_dir=receipts_dir,
        )

    raise ValueError(f"unsupported anchor_mode: {mode}")
