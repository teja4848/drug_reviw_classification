# test_model_inference.py
"""
Test script to verify both saved models can be loaded and used for inference.
"""

import joblib
import pandas as pd
from pathlib import Path


def test_model(model_path):
    """Load model and test inference."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_path}")
    print('='*60)
    
    try:
        # Load model
        print("Loading model...", end=" ")
        model = joblib.load(model_path)
        print("✓")
        
        if hasattr(model, 'named_steps'):
            print(f"Pipeline steps: {list(model.named_steps.keys())}")
        
        # Create sample data
        sample_data = pd.DataFrame({
            'longitude': [-122.23, -118.25, -117.81],
            'latitude': [37.88, 34.05, 33.68],
            'housing_median_age': [41.0, 28.0, 15.0],
            'total_rooms': [880.0, 2000.0, 1500.0],
            'total_bedrooms': [129.0, 400.0, 300.0],
            'population': [322.0, 1000.0, 800.0],
            'households': [126.0, 380.0, 280.0],
            'median_income': [8.3252, 5.6431, 3.8462],
            'ocean_proximity': ['NEAR BAY', 'NEAR OCEAN', 'INLAND']
        })
        
        # Predict
        print("Running inference...", end=" ")
        predictions = model.predict(sample_data)
        print("✓")
        
        # Show results
        print("\nPredictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"  Sample {i}: ${pred:,.2f}")
        
        print(f"\n✓ {Path(model_path).name} - SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return False


def main():
    """Test both models."""
    models = [
        "models/global_best_model.pkl",
        "models/global_best_model_optuna.pkl"
    ]
    
    print("Testing both models...")
    results = [test_model(m) for m in models]
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {sum(results)}/{len(results)} models passed")
    print('='*60)
    
    return all(results)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)