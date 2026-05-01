#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
FedTracker WebUI CLI Tool
A command-line interface for managing the WebUI
"""

import argparse
import subprocess
import sys
import os


def cmd_start(args):
    """Start the WebUI server"""
    print("🚀 Starting FedTracker WebUI...")

    port = args.port or 7860
    host = args.host or "0.0.0.0"
    share = args.share or False

    cmd = [sys.executable, "app.py", "--port", str(port), "--host", host]
    if share:
        cmd.append("--share")

    print(f"   Port: {port}")
    print(f"   Host: {host}")
    print(f"   Share: {'Yes' if share else 'No'}")
    print()

    os.chdir(os.path.dirname(__file__))
    subprocess.run(cmd)


def cmd_test(args):
    """Run configuration tests"""
    print("🧪 Running configuration tests...")
    subprocess.run([sys.executable, "test_config.py"])


def cmd_check(args):
    """Check dependencies and models"""
    print("🔍 Checking environment...")

    # Check Python version
    print(f"\n✅ Python version: {sys.version.split()[0]}")

    # Check dependencies
    print("\n📦 Checking dependencies:")
    deps = ["gradio", "torch", "PIL", "numpy", "matplotlib"]
    for dep in deps:
        try:
            __import__(dep if dep != "PIL" else "PIL")
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} (missing)")

    # Check models
    print("\n🤖 Checking models:")
    result_dir = os.path.join(os.path.dirname(__file__), "..", "result")
    if os.path.exists(result_dir):
        models = [
            d
            for d in os.listdir(result_dir)
            if os.path.isdir(os.path.join(result_dir, d))
        ]
        if models:
            print(f"   Found {len(models)} model(s):")
            for model in models:
                model_path = os.path.join(result_dir, model)
                has_final = os.path.exists(os.path.join(model_path, "model_final.pth"))
                has_trace = os.path.exists(os.path.join(model_path, "trace_data"))
                print(f"   - {model}")
                print(f"     - model_final.pth: {'✅' if has_final else '❌'}")
                print(f"     - trace_data: {'✅' if has_trace else '❌'}")
        else:
            print("   ⚠️ No models found in result/ directory")
    else:
        print("   ⚠️ result/ directory not found")


def cmd_install(args):
    """Install dependencies"""
    print("📥 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def main():
    parser = argparse.ArgumentParser(
        description="FedTracker WebUI CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start                    # Start server on default port (7860)
  %(prog)s start --port 8080        # Start server on port 8080
  %(prog)s start --share            # Start server with public share link
  %(prog)s test                     # Run configuration tests
  %(prog)s check                    # Check environment and models
  %(prog)s install                  # Install dependencies
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start the WebUI server")
    start_parser.add_argument("--port", type=int, help="Port number (default: 7860)")
    start_parser.add_argument(
        "--host", type=str, help="Host address (default: 0.0.0.0)"
    )
    start_parser.add_argument(
        "--share", action="store_true", help="Create public share link"
    )

    # Test command
    subparsers.add_parser("test", help="Run configuration tests")

    # Check command
    subparsers.add_parser("check", help="Check environment and models")

    # Install command
    subparsers.add_parser("install", help="Install dependencies")

    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "check":
        cmd_check(args)
    elif args.command == "install":
        cmd_install(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
