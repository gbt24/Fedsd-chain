#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Quick start helper script for FedTracker WebUI
"""

import subprocess
import sys
import os


def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")

    required = ["gradio", "torch", "PIL"]
    missing = []

    for package in required:
        try:
            __import__(package if package != "PIL" else "PIL")
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (missing)")
            missing.append(package)

    if missing:
        print("\n📦 Installing missing dependencies...")
        for package in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    return True


def check_model_files():
    """Check if model files exist"""
    print("\n🔍 Checking model files...")
    result_dir = os.path.join(os.path.dirname(__file__), "..", "result")

    if not os.path.exists(result_dir):
        print(f"  ⚠️ No result directory found at {result_dir}")
        return False

    models = []
    for name in os.listdir(result_dir):
        model_path = os.path.join(result_dir, name)
        model_file = os.path.join(model_path, "model_final.pth")
        if os.path.isdir(model_path) and os.path.exists(model_file):
            models.append(name)

    if models:
        print(f"  ✅ Found {len(models)} model(s):")
        for model in models:
            print(f"     - {model}")
        return True
    else:
        print("  ⚠️ No trained models found in result/ directory")
        return False


def main():
    print("=" * 60)
    print("FedTracker WebUI - Quick Start")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed")
        return

    # Check model files
    check_model_files()

    # Start WebUI
    print("\n🚀 Starting WebUI...")
    print("=" * 60)

    os.chdir(os.path.dirname(__file__))
    subprocess.call([sys.executable, "app.py", "--port", "7860", "--host", "0.0.0.0"])


if __name__ == "__main__":
    main()
