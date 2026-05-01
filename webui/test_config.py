# -*- coding: UTF-8 -*-
"""
Test configuration for WebUI
Tests basic functionality without running the full UI
"""

import os
import sys

# Add parent directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
webui_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, webui_dir)

try:
    from modules.utils import find_model_dirs, has_trace_data, get_default_output_dir
except ImportError:
    # Direct imports for testing
    exec(open(os.path.join(os.path.dirname(__file__), "modules", "utils.py")).read())


def test_module_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    try:
        from webui.modules import utils
        from webui.modules import generation
        from webui.modules import tracing

        print("✅ All modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_find_models():
    """Test model directory detection"""
    print("\nTesting model directory detection...")
    model_dirs = find_model_dirs("./result")
    print(f"Found {len(model_dirs)} model directories")
    if model_dirs:
        print(f"Models: {model_dirs}")
        for model in model_dirs:
            trace_available = has_trace_data(f"./result/{model}")
            print(f"  - {model}: trace_data={'✅' if trace_available else '❌'}")
    return len(model_dirs) > 0


def test_output_dir():
    """Test output directory creation"""
    print("\nTesting output directory...")
    output_dir = get_default_output_dir()
    print(f"Output directory: {output_dir}")

    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ Output directory created/exists")
        return True
    except Exception as e:
        print(f"❌ Failed to create output directory: {e}")
        return False


def test_css_loading():
    """Test that CSS is properly defined"""
    print("\nTesting CSS configuration...")
    try:
        from webui.app import CSS

        print(f"✅ CSS loaded ({len(CSS)} characters)")
        return True
    except Exception as e:
        print(f"❌ CSS loading failed: {e}")
        return False


def run_tests():
    """Run all tests"""
    print("=" * 60)
    print("FedTracker WebUI - Configuration Test")
    print("=" * 60)

    results = []
    results.append(test_module_imports())
    results.append(test_find_models())
    results.append(test_output_dir())
    results.append(test_css_loading())

    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)

    if all(results):
        print("\n✅ All tests passed! WebUI is ready to use.")
        print("\nTo start the WebUI, run:")
        print("  cd webui")
        print("  python app.py --port 7860")
    else:
        print("\n⚠️ Some tests failed. Please check the configuration.")


if __name__ == "__main__":
    run_tests()
