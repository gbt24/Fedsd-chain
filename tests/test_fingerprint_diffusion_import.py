# -*- coding: UTF-8 -*-
import unittest


class FingerprintDiffusionImportTest(unittest.TestCase):
    def test_module_imports_without_geneal_for_non_generation_usage(self):
        import watermark.fingerprint_diffusion as fingerprint_diffusion

        self.assertTrue(hasattr(fingerprint_diffusion, "calculate_local_grad"))
        self.assertTrue(hasattr(fingerprint_diffusion, "extracting_fingerprints"))


if __name__ == "__main__":
    unittest.main()
