# -*- coding: UTF-8 -*-
import os
import tempfile
import unittest


class Args:
    pass


class AnchorFactoryTest(unittest.TestCase):
    def _args(self, **values):
        args = Args()
        defaults = {
            "enable_anchor": False,
            "anchor_mode": "mock",
            "anchor_path": None,
            "anchor_rpc": None,
            "anchor_contract": None,
        }
        defaults.update(values)
        for key, value in defaults.items():
            setattr(args, key, value)
        return args

    def test_disabled_anchor_returns_none(self):
        from blockchain.anchor_factory import create_anchor_client_from_args

        self.assertIsNone(create_anchor_client_from_args(self._args()))

    def test_mock_anchor_uses_default_path_under_save_dir(self):
        from blockchain.anchor_factory import create_anchor_client_from_args
        from blockchain.mock_anchor import MockAnchorClient

        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._args(enable_anchor=True, save_dir=tmpdir)
            client = create_anchor_client_from_args(args)

        self.assertIsInstance(client, MockAnchorClient)
        self.assertTrue(client.anchor_path.endswith(os.path.join("blockchain", "anchors.jsonl")))

    def test_evm_anchor_requires_rpc_and_contract(self):
        from blockchain.anchor_factory import create_anchor_client_from_args

        args = self._args(enable_anchor=True, anchor_mode="evm")
        with self.assertRaises(ValueError):
            create_anchor_client_from_args(args)


if __name__ == "__main__":
    unittest.main()
