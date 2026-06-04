# -*- coding: UTF-8 -*-
import json
import os
import tempfile
import unittest

from tests.test_evm_anchor_mocked import FakeHexBytes, FakeWeb3


class DeployEVMAnchorTest(unittest.TestCase):
    def test_load_compiled_contract_reads_abi_and_bytecode(self):
        from blockchain.deploy_evm_anchor import load_compiled_contract

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = os.path.join(tmpdir, "EvidenceAnchor.json")
            with open(artifact_path, "w") as f:
                json.dump({"abi": [{"type": "constructor"}], "bytecode": "0x6000"}, f)

            artifact = load_compiled_contract(artifact_path)

        self.assertEqual(artifact["bytecode"], "0x6000")
        self.assertEqual(artifact["abi"][0]["type"], "constructor")

    def test_deploy_contract_saves_deployment_receipt(self):
        from blockchain.deploy_evm_anchor import deploy_contract

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "anchor_deployment.json")
            web3_client = DeployWeb3()

            deployment = deploy_contract(
                web3_client=web3_client,
                abi=[],
                bytecode="0x6000",
                private_key="0x" + "2" * 64,
                out_path=out_path,
            )

            with open(out_path, "r") as f:
                saved = json.load(f)

        self.assertEqual(deployment["contract_address"], "0x" + "3" * 40)
        self.assertEqual(saved["deployment_tx_hash"], "0x" + "8" * 64)
        self.assertEqual(saved["deployment_block_number"], 123)
        self.assertEqual(saved["deployment_block_timestamp"], 1710000010)
        self.assertEqual(saved["abi"], [])
        self.assertEqual(web3_client.sent_raw_transaction, b"signed")


class DeployConstructor:
    def build_transaction(self, tx):
        built = dict(tx)
        built["data"] = "0x6000"
        return built


class DeployContractFactory:
    def constructor(self):
        return DeployConstructor()


class DeployEth(FakeWeb3().eth.__class__):
    def contract(self, abi, bytecode):
        return DeployContractFactory()

    def send_raw_transaction(self, raw_transaction):
        self.sent_raw_transaction = raw_transaction
        return FakeHexBytes("0x" + "8" * 64)

    def wait_for_transaction_receipt(self, tx_hash, timeout=120, poll_latency=0.1):
        return {
            "transactionHash": FakeHexBytes("0x" + "8" * 64),
            "contractAddress": "0x" + "3" * 40,
            "blockNumber": 123,
            "status": 1,
            "gasUsed": 55000,
        }

    def get_block(self, block_number):
        return {"timestamp": 1710000010}


class DeployWeb3(FakeWeb3):
    def __init__(self):
        self.eth = DeployEth()


if __name__ == "__main__":
    unittest.main()
