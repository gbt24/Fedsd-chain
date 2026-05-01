#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Simple test script to verify WebUI modules are working correctly.
Run this from the project root: python webui/test_modules.py
"""

import os
import sys

print("=" * 60)
print("FedTracker WebUI - Module Verification")
print("=" * 60)

# Test 1: Check Python version
print(f"\n✅ Python version: {sys.version.split()[0]}")

# Add project root to path if running as 'python webui/test_modules.py'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Test 2: Import modules
print("\n📦 Testing module imports...")

# Test core project modules
try:
    from utils.utils import load_args

    print("  ✅ utils.utils")
except ImportError as e:
    print(f"  ⚠️  utils.utils: {e} (requires torch)")

try:
    from utils.simple_unet import ClassConditionalUNet

    print("  ✅ utils.simple_unet")
except ImportError as e:
    print(f"  ⚠️  utils.simple_unet: {e} (requires torch)")

try:
    from utils.simple_diffusion import SimpleDiffusion

    print("  ✅ utils.simple_diffusion")
except ImportError as e:
    print(f"  ⚠️  utils.simple_diffusion: {e} (requires torch)")

# Test WebUI modules
try:
    from modules.utils import find_model_dirs, has_trace_data, get_default_output_dir

    print("  ✅ modules.utils")
except ImportError as e:
    print(f"  ❌ modules.utils: {e}")
    sys.exit(1)

try:
    from modules.generation import generate_images, save_images

    print("  ✅ modules.generation")
except ImportError as e:
    print(f"  ⚠️  modules.generation: {e} (requires torch)")

try:
    from modules.tracing import simulate_client_leak, identify_owner, get_client_list

    print("  ✅ modules.tracing")
except ImportError as e:
    print(f"  ⚠️  modules.tracing: {e} (requires torch)")

# Test 3: Find models
print("\n🤖 Testing model discovery...")
from modules.utils import find_model_dirs, has_trace_data

models = find_model_dirs("./result")
if models:
    print(f"  ✅ Found {len(models)} model(s):")
    for model in models:
        has_trace = has_trace_data(f"./result/{model}")
        print(f"     - {model} (trace_data: {'✅' if has_trace else '❌'})")
else:
    print("  ⚠️  No models found in ./result/")
    print("     This is normal if you haven't trained models yet.")

# Test 4: Check output directory
print("\n📁 Testing output directory...")
from modules.utils import get_default_output_dir

output_dir = get_default_output_dir()
try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"  ✅ Output directory created: {output_dir}")
except Exception as e:
    print(f"  ❌ Failed to create output directory: {e}")

# Test 5: Check optional dependencies
print("\n📦 Checking dependencies...")
deps = {
    "torch": "PyTorch (required for generation)",
    "PIL": "Pillow (required for image processing)",
    "numpy": "NumPy (required for numerics)",
    "gradio": "Gradio (required for WebUI)",
}
has_missing = False
for dep, desc in deps.items():
    try:
        __import__(dep)
        print(f"  ✅ {dep} - {desc}")
    except ImportError:
        print(f"  ❌ {dep} - {desc} (MISSING)")
        has_missing = True

print("\n" + "=" * 60)
if has_missing:
    print("⚠️  Some dependencies are missing.")
    print("   Install with: pip install -r webui/requirements.txt")
else:
    print("✅ All tests passed! WebUI is ready to run.")
print("=" * 60)

print("\nTo start the WebUI:")
print("  cd webui")
print("  python app.py --port 7860 --host 0.0.0.0")
print("\nFor public sharing:")
print("  python app.py --port 7860 --share")
