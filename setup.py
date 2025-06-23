#!/usr/bin/env python3
"""
Setup Script for XRP Price Forecasting Model
=============================================

This script helps set up the environment and dependencies for the XRP price forecasting model.
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    try:
        # Try to install packages
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, check=True)
        
        print("✅ All packages installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ Failed to install packages")
        print("Error output:")
        print(e.stderr)
        
        # Try alternative installation method
        print("\n🔄 Trying alternative installation method...")
        packages = [
            "numpy", "pandas", "scikit-learn", 
            "tensorflow", "matplotlib", "seaborn"
        ]
        
        failed_packages = []
        for package in packages:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True, check=True)
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                failed_packages.append(package)
        
        if failed_packages:
            print(f"\n⚠️  Some packages failed to install: {failed_packages}")
            print("You may need to install them manually or use a different environment")
            return False
        else:
            print("✅ All packages installed via alternative method!")
            return True

def check_dataset():
    """Check if dataset files exist"""
    print("\n📁 Checking dataset files...")
    
    if not os.path.exists("dataset"):
        print("❌ Dataset folder not found")
        return False
    
    dataset_files = [f for f in os.listdir("dataset") if f.endswith(".json")]
    if not dataset_files:
        print("❌ No dataset files found in dataset folder")
        return False
    
    print(f"✅ Found {len(dataset_files)} dataset files")
    for file in dataset_files:
        print(f"   - {file}")
    
    return True

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🔍 Testing package imports...")
    
    required_packages = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("sklearn", None),
        ("tensorflow", "tf"),
        ("matplotlib", None),
    ]
    
    failed_imports = []
    
    for package, alias in required_packages:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {str(e)}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️  Failed to import: {failed_imports}")
        return False
    else:
        print("✅ All packages imported successfully!")
        return True

def run_structure_test():
    """Run the model structure test"""
    print("\n🧪 Running model structure test...")
    
    try:
        result = subprocess.run([
            sys.executable, "test_model.py"
        ], capture_output=True, text=True, check=True)
        
        print("✅ Model structure test passed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ Model structure test failed")
        print("Error output:")
        print(e.stderr)
        return False

def main():
    """Main setup function"""
    print("🚀 XRP Price Forecasting Model - Setup")
    print("=" * 50)
    
    # Run all setup checks
    steps = [
        ("Python Version Check", check_python_version),
        ("Install Requirements", install_requirements),
        ("Check Dataset", check_dataset),
        ("Test Imports", test_imports),
        ("Structure Test", run_structure_test)
    ]
    
    passed = 0
    total = len(steps)
    
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        if step_func():
            passed += 1
        else:
            print(f"❌ {step_name} failed")
            break
    
    print("\n" + "=" * 50)
    print(f"Setup Results: {passed}/{total} steps completed")
    
    if passed == total:
        print("🎉 Setup completed successfully!")
        print("\n📚 Next Steps:")
        print("1. Train the model:")
        print("   python example_usage.py")
        print("\n2. Make predictions:")
        print("   python predict.py <recent_data_file>")
        print("\n3. For quick testing:")
        print("   python model.py")
        
    else:
        print("❌ Setup incomplete. Please resolve the issues above.")
        
        if passed >= 2:  # If we got past installation
            print("\n💡 You can still try to run the model manually:")
            print("   python test_model.py  # Test structure")
            print("   python model.py       # Train model")

if __name__ == "__main__":
    main()
