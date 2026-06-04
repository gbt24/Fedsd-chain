# -*- coding: UTF-8 -*-
import json
import os

from blockchain.anchor_client import AnchorClient, build_anchor_record

try:
    from web3 import Web3
except ModuleNotFoundError:
    Web3 = None


def _bytes32_hex(value):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"expected 64-character hex hash, got {value!r}")
    int(value, 16)
    return "0x" + value


def build_contract_payload(block, payload_root_hash):
    record = build_anchor_record(block, payload_root_hash)
    return {
        "anchor_id": _bytes32_hex(record["anchor_id"]),
        "run_id_hash": _bytes32_hex(record["run_id_hash"]),
        "event_type_hash": _bytes32_hex(record["event_type_hash"]),
        "local_block_index": record["local_block_index"],
        "local_block_hash": _bytes32_hex(record["local_block_hash"]),
        "payload_root_hash": _bytes32_hex(record["payload_root_hash"]),
        "previous_local_block_hash": _bytes32_hex(record["previous_local_block_hash"]),
    }


class EVMAnchorClient(AnchorClient):
    def __init__(
        self,
        rpc_url,
        contract_address,
        private_key=None,
        contract=None,
        abi=None,
        web3_client=None,
        receipts_dir=None,
        timeout=120,
    ):
        if web3_client is None and Web3 is None:
            raise RuntimeError("web3 is required for EVM anchoring; install web3>=6.0")
        self.web3 = web3_client or Web3(Web3.HTTPProvider(rpc_url))
        self.contract_address = contract_address
        self.private_key = private_key or os.environ.get("FDMOT_ANCHOR_PRIVATE_KEY")
        if contract is None and not abi and not self.private_key:
            raise RuntimeError(
                "web3 EVM anchoring requires FDMOT_ANCHOR_PRIVATE_KEY or private_key"
            )
        if contract is None and abi:
            contract = self.web3.eth.contract(address=contract_address, abi=abi)
        self.contract = contract
        self.receipts_dir = receipts_dir
        self.timeout = timeout
        if self.receipts_dir:
            os.makedirs(self.receipts_dir, exist_ok=True)

    def _account(self):
        if not self.private_key:
            raise RuntimeError("FDMOT_ANCHOR_PRIVATE_KEY or private_key is required")
        return self.web3.eth.account.from_key(self.private_key)

    def _tx_hash_hex(self, value):
        text = value.hex() if hasattr(value, "hex") else str(value)
        return text if text.startswith("0x") else "0x" + text

    def _bytes32_value(self, value):
        text = value.hex() if hasattr(value, "hex") else str(value)
        return text[2:] if text.startswith("0x") else text

    def _receipt_path(self, anchor_id):
        if not self.receipts_dir:
            return None
        return os.path.join(self.receipts_dir, f"{anchor_id}.json")

    def _save_receipt(self, receipt):
        receipt_path = self._receipt_path(receipt["anchor_id"])
        if receipt_path:
            with open(receipt_path, "w") as f:
                json.dump(receipt, f, indent=2)

    def anchor_block(self, block, payload_root_hash):
        if self.contract is None:
            raise RuntimeError("contract binding is required for EVM anchoring")
        payload = build_contract_payload(block, payload_root_hash)
        account = self._account()
        function = self.contract.functions.anchorEvidence(
            payload["anchor_id"],
            payload["run_id_hash"],
            payload["event_type_hash"],
            payload["local_block_index"],
            payload["local_block_hash"],
            payload["payload_root_hash"],
            payload["previous_local_block_hash"],
        )
        tx = function.build_transaction(
            {
                "from": account.address,
                "nonce": self.web3.eth.get_transaction_count(account.address),
                "chainId": self.web3.eth.chain_id,
            }
        )
        signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.raw_transaction)
        chain_receipt = self.web3.eth.wait_for_transaction_receipt(
            tx_hash, timeout=self.timeout
        )
        block_data = self.web3.eth.get_block(chain_receipt["blockNumber"])
        receipt = {
            "anchor_mode": "evm",
            "anchored": True,
            "anchor_id": payload["anchor_id"][2:],
            "chain_id": self.web3.eth.chain_id,
            "contract_address": self.contract_address,
            "tx_hash": self._tx_hash_hex(chain_receipt["transactionHash"]),
            "block_number": chain_receipt["blockNumber"],
            "block_timestamp": block_data["timestamp"],
            "status": chain_receipt.get("status"),
            "gas_used": chain_receipt.get("gasUsed"),
            "local_block_hash": payload["local_block_hash"][2:],
            "payload_root_hash": payload["payload_root_hash"][2:],
        }
        self._save_receipt(receipt)
        return receipt

    def verify_anchor(self, block, payload_root_hash):
        if self.contract is None:
            return {"verified": False, "reason": "contract binding is required"}
        payload = build_contract_payload(block, payload_root_hash)
        anchor = self.contract.functions.getAnchor(payload["anchor_id"]).call()
        exists = bool(anchor[0])
        if not exists:
            return {
                "verified": False,
                "reason": "on-chain anchor not found",
                "anchor_id": payload["anchor_id"][2:],
                "anchor_mode": "evm",
            }
        local_block_hash = self._bytes32_value(anchor[4])
        payload_root = self._bytes32_value(anchor[5])
        verified = (
            local_block_hash == payload["local_block_hash"][2:]
            and payload_root == payload["payload_root_hash"][2:]
        )
        return {
            "verified": verified,
            "reason": "on-chain anchor matches" if verified else "on-chain anchor mismatch",
            "anchor_id": payload["anchor_id"][2:],
            "anchor_mode": "evm",
            "chain_id": self.web3.eth.chain_id,
            "contract_address": self.contract_address,
            "onchain_local_block_hash_match": local_block_hash == payload["local_block_hash"][2:],
            "onchain_payload_root_hash_match": payload_root == payload["payload_root_hash"][2:],
        }
