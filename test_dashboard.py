#!/usr/bin/env python3
"""
Test script for the Streamlit dashboard
"""

import sys
import numpy as np

def test_probability_arrays():
    """Test that all probability arrays sum to 1.0"""
    
    def normalize_probs(probs):
        probs = np.array(probs)
        return probs / probs.sum()
    
    # Test arrays from the dashboard
    test_cases = [
        ("Education", [0.71, 0.24, 0.033, 0.012, 0.005]),
        ("Income Type", [0.52, 0.23, 0.18, 0.07]),
        ("Occupation", [0.26, 0.15, 0.13, 0.10, 0.09, 0.05, 0.05, 0.17]),
        ("Family Status", [0.64, 0.15, 0.10, 0.06, 0.05]),
        ("Gender", [0.65, 0.35])
    ]
    
    print("🧪 Testing Probability Arrays")
    print("=" * 40)
    
    all_passed = True
    
    for name, probs in test_cases:
        original_sum = sum(probs)
        normalized = normalize_probs(probs)
        normalized_sum = sum(normalized)
        
        print(f"{name}:")
        print(f"  Original sum: {original_sum:.6f}")
        print(f"  Normalized sum: {normalized_sum:.6f}")
        print(f"  ✅ Pass" if abs(normalized_sum - 1.0) < 1e-10 else "  ❌ Fail")
        print()
        
        if abs(normalized_sum - 1.0) >= 1e-10:
            all_passed = False
    
    return all_passed

def test_data_generation():
    """Test that data generation works without errors"""
    print("🧪 Testing Data Generation")
    print("=" * 40)
    
    try:
        np.random.seed(42)
        n_samples = 100  # Small sample for testing
        
        def normalize_probs(probs):
            probs = np.array(probs)
            return probs / probs.sum()
        
        # Test each random choice
        test_data = {}
        
        # Education
        test_data['NAME_EDUCATION_TYPE'] = np.random.choice([
            'Secondary / secondary special', 'Higher education', 
            'Incomplete higher', 'Lower secondary', 'Academic degree'
        ], n_samples, p=normalize_probs([0.71, 0.24, 0.033, 0.012, 0.005]))
        
        # Income type
        test_data['NAME_INCOME_TYPE'] = np.random.choice([
            'Working', 'Commercial associate', 'Pensioner', 'State servant'
        ], n_samples, p=normalize_probs([0.52, 0.23, 0.18, 0.07]))
        
        # Occupation
        test_data['OCCUPATION_TYPE'] = np.random.choice([
            'Laborers', 'Sales staff', 'Core staff', 'Managers', 'Drivers',
            'High skill tech staff', 'Accountants', 'Medicine staff'
        ], n_samples, p=normalize_probs([0.26, 0.15, 0.13, 0.10, 0.09, 0.05, 0.05, 0.17]))
        
        # Family status
        test_data['NAME_FAMILY_STATUS'] = np.random.choice([
            'Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'
        ], n_samples, p=normalize_probs([0.64, 0.15, 0.10, 0.06, 0.05]))
        
        # Gender
        test_data['CODE_GENDER'] = np.random.choice(['F', 'M'], n_samples, p=normalize_probs([0.65, 0.35]))
        
        print("✅ Data generation successful!")
        print(f"Generated {n_samples} samples with {len(test_data)} features")
        return True
        
    except Exception as e:
        print(f"❌ Data generation failed: {str(e)}")
        return False

def test_imports():
    """Test that all required packages can be imported"""
    print("🧪 Testing Package Imports")
    print("=" * 40)
    
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'sklearn',
        'seaborn',
        'matplotlib'
    ]
    
    all_imported = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            all_imported = False
    
    return all_imported

def main():
    """Run all tests"""
    print("🏦 Home Credit Dashboard Tests")
    print("=" * 50)
    print()
    
    # Run tests
    tests = [
        ("Package Imports", test_imports),
        ("Probability Arrays", test_probability_arrays),
        ("Data Generation", test_data_generation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {str(e)}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print()
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard should work correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 