# -*- coding: UTF-8 -*-
import argparse
import json
import os
import subprocess

try:
    from web3 import Web3
except ModuleNotFoundError:
    Web3 = None


def load_compiled_contract(artifact_path):
    with open(artifact_path, "r") as f:
        artifact = json.load(f)
    abi = artifact.get("abi")
    bytecode = artifact.get("bytecode") or artifact.get("bin")
    if isinstance(bytecode, dict):
        bytecode = bytecode.get("object")
    if not abi or not bytecode:
        raise ValueError("compiled contract artifact must include abi and bytecode")
    if not bytecode.startswith("0x"):
        bytecode = "0x" + bytecode
    return {"abi": abi, "bytecode": bytecode}


def compile_with_solc(contract_path, contract_name="EvidenceAnchor"):
    result = subprocess.run(
        ["solc", "--combined-json", "abi,bin", contract_path],
        check=True,
        capture_output=True,
        text=True,
    )
    compiled = json.loads(result.stdout)
    target_suffix = f":{contract_name}"
    for name, artifact in compiled.get("contracts", {}).items():
        if name.endswith(target_suffix):
            abi = artifact["abi"]
            if isinstance(abi, str):
                abi = json.loads(abi)
            return {"abi": abi, "bytecode": "0x" + artifact["bin"]}
    raise ValueError(f"contract {contract_name} not found in solc output")


def _tx_hash_hex(value):
    text = value.hex() if hasattr(value, "hex") else str(value)
    return text if text.startswith("0x") else "0x" + text


def deploy_contract(web3_client, abi, bytecode, private_key, out_path, timeout=120):
    account = web3_client.eth.account.from_key(private_key)
    contract_factory = web3_client.eth.contract(abi=abi, bytecode=bytecode)
    tx = contract_factory.constructor().build_transaction(
        {
            "from": account.address,
            "nonce": web3_client.eth.get_transaction_count(account.address),
            "chainId": web3_client.eth.chain_id,
        }
    )
    signed_tx = web3_client.eth.account.sign_transaction(tx, private_key)
    tx_hash = web3_client.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt = web3_client.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
    block_data = web3_client.eth.get_block(receipt["blockNumber"])
    deployment = {
        "chain_id": web3_client.eth.chain_id,
        "contract_address": receipt["contractAddress"],
        "abi": abi,
        "deployment_tx_hash": _tx_hash_hex(receipt["transactionHash"]),
        "deployment_block_number": receipt["blockNumber"],
        "deployment_block_timestamp": block_data["timestamp"],
        "status": receipt.get("status"),
        "gas_used": receipt.get("gasUsed"),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(deployment, f, indent=2)
    return deployment


def main():
    parser = argparse.ArgumentParser(description="Deploy EvidenceAnchor to an EVM network")
    parser.add_argument("--rpc", required=True, help="Sepolia/Polygon Amoy RPC URL")
    parser.add_argument(
        "--artifact",
        default=None,
        help="compiled contract JSON with abi and bytecode",
    )
    parser.add_argument(
        "--contract",
        default=os.path.join(os.path.dirname(__file__), "contracts", "EvidenceAnchor.sol"),
        help="Solidity contract path used when --artifact is omitted and solc is installed",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="deployment receipt JSON output path",
    )
    parser.add_argument("--timeout", type=int, default=120, help="receipt wait timeout")
    args = parser.parse_args()

    if Web3 is None:
        raise RuntimeError("web3 is required; install dependencies with pip install -r requirements.txt")
    private_key = os.environ.get("FDMOT_ANCHOR_PRIVATE_KEY")
    if not private_key:
        raise RuntimeError("set FDMOT_ANCHOR_PRIVATE_KEY before deploying")

    artifact = load_compiled_contract(args.artifact) if args.artifact else compile_with_solc(args.contract)
    web3_client = Web3(Web3.HTTPProvider(args.rpc))
    deployment = deploy_contract(
        web3_client=web3_client,
        abi=artifact["abi"],
        bytecode=artifact["bytecode"],
        private_key=private_key,
        out_path=args.out,
        timeout=args.timeout,
    )
    print(json.dumps(deployment, indent=2))


if __name__ == "__main__":
    main()
