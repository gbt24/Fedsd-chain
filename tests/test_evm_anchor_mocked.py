# -*- coding: UTF-8 -*-
import os
import tempfile
import unittest
from unittest.mock import patch


class EVMAnchorTest(unittest.TestCase):
    def _sample_block(self):
        return {
            "block_index": 7,
            "event_type": "final_model_commit",
            "run_id": "run-1",
            "payload": {"model_hash": "d" * 64},
            "previous_block_hash": "b" * 64,
            "block_hash": "c" * 64,
        }

    def test_evm_anchor_module_import_does_not_require_web3(self):
        from blockchain.evm_anchor import build_contract_payload

        self.assertTrue(callable(build_contract_payload))

    def test_build_contract_payload_uses_bytes32_hex_values(self):
        from blockchain.evm_anchor import build_contract_payload

        payload = build_contract_payload(self._sample_block(), "d" * 64)

        self.assertEqual(payload["local_block_index"], 7)
        for key in (
            "anchor_id",
            "run_id_hash",
            "event_type_hash",
            "local_block_hash",
            "payload_root_hash",
            "previous_local_block_hash",
        ):
            self.assertTrue(payload[key].startswith("0x"), key)
            self.assertEqual(len(payload[key]), 66, key)

    def test_invalid_bytes32_hash_is_rejected(self):
        from blockchain.evm_anchor import build_contract_payload

        with self.assertRaises(ValueError):
            build_contract_payload(self._sample_block(), "not-a-hash")

    def test_missing_web3_dependency_has_clear_error(self):
        from blockchain.evm_anchor import EVMAnchorClient

        with patch("blockchain.evm_anchor.Web3", None):
            with self.assertRaises(RuntimeError) as ctx:
                EVMAnchorClient(rpc_url="https://example.invalid", contract_address="0x" + "1" * 40)

        self.assertIn("web3", str(ctx.exception))

    def test_anchor_block_builds_signed_transaction_and_saves_receipt(self):
        from blockchain.evm_anchor import EVMAnchorClient

        with tempfile.TemporaryDirectory() as tmpdir:
            web3_client = FakeWeb3()
            contract = FakeContract()
            client = EVMAnchorClient(
                rpc_url="https://example.invalid",
                contract_address="0x" + "1" * 40,
                private_key="0x" + "2" * 64,
                contract=contract,
                web3_client=web3_client,
                receipts_dir=tmpdir,
            )

            receipt = client.anchor_block(self._sample_block(), "d" * 64)

            receipt_path = os.path.join(tmpdir, f"{receipt['anchor_id']}.json")
            self.assertTrue(os.path.exists(receipt_path))
            self.assertEqual(receipt["anchor_mode"], "evm")
            self.assertEqual(receipt["chain_id"], 11155111)
            self.assertEqual(receipt["block_number"], 100)
            self.assertEqual(receipt["block_timestamp"], 1710000000)
            self.assertEqual(receipt["status"], 1)
            self.assertEqual(web3_client.sent_raw_transaction, b"signed")

    def test_verify_anchor_returns_chain_metadata(self):
        from blockchain.evm_anchor import EVMAnchorClient

        web3_client = FakeWeb3()
        contract = FakeContract()
        client = EVMAnchorClient(
            rpc_url="https://example.invalid",
            contract_address="0x" + "1" * 40,
            private_key="0x" + "2" * 64,
            contract=contract,
            web3_client=web3_client,
        )

        result = client.verify_anchor(self._sample_block(), "d" * 64)

        self.assertTrue(result["verified"])
        self.assertEqual(result["chain_id"], 11155111)
        self.assertEqual(result["contract_address"], "0x" + "1" * 40)


class FakeHexBytes:
    def __init__(self, value):
        self.value = value

    def hex(self):
        return self.value


class FakeAccount:
    address = "0x" + "a" * 40

    def from_key(self, private_key):
        return self

    def sign_transaction(self, tx, private_key):
        self.signed_tx = tx
        return type("SignedTx", (), {"raw_transaction": b"signed"})()


class FakeEth:
    chain_id = 11155111

    def __init__(self):
        self.account = FakeAccount()

    def get_transaction_count(self, address):
        return 7

    def send_raw_transaction(self, raw_transaction):
        self.sent_raw_transaction = raw_transaction
        return FakeHexBytes("0x" + "9" * 64)

    def wait_for_transaction_receipt(self, tx_hash, timeout=120, poll_latency=0.1):
        return {
            "transactionHash": FakeHexBytes("0x" + "9" * 64),
            "blockNumber": 100,
            "status": 1,
            "gasUsed": 25000,
        }

    def get_block(self, block_number):
        return {"timestamp": 1710000000}


class FakeWeb3:
    def __init__(self):
        self.eth = FakeEth()

    @property
    def sent_raw_transaction(self):
        return self.eth.sent_raw_transaction


class FakeAnchorFunction:
    def build_transaction(self, tx):
        built = dict(tx)
        built["data"] = "0xanchor"
        return built


class FakeFunctions:
    def anchorEvidence(self, *args):
        return FakeAnchorFunction()

    def getAnchor(self, anchor_id):
        return type(
            "Call",
            (),
            {
                "call": lambda self: (
                    True,
                    bytes.fromhex("1" * 64),
                    bytes.fromhex("2" * 64),
                    7,
                    bytes.fromhex("c" * 64),
                    bytes.fromhex("d" * 64),
                    bytes.fromhex("b" * 64),
                    1710000000,
                    "0x" + "a" * 40,
                )
            },
        )()


class FakeContract:
    def __init__(self):
        self.functions = FakeFunctions()


if __name__ == "__main__":
    unittest.main()
