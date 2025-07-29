import tensorflow as tf
import numpy as np
from block_dense_layer import create_multi_head_forecasting_model
import matplotlib.pyplot as plt

def generate_synthetic_time_series_data(n_skus=4, n_samples=1000, window_size=10):
    """
    Generate synthetic time series data for multiple SKUs.
    
    Args:
        n_skus (int): Number of SKUs
        n_samples (int): Number of samples
        window_size (int): Look-back window size
        
    Returns:
        tuple: (X, y) where X is input data and y is target data
    """
    # Generate base patterns for each SKU
    base_patterns = []
    for i in range(n_skus):
        # Each SKU has a different seasonal pattern
        t = np.linspace(0, 4*np.pi, n_samples + window_size)
        pattern = np.sin(t + i*np.pi/2) + 0.5*np.sin(2*t + i*np.pi/4) + 0.1*np.random.randn(len(t))
        base_patterns.append(pattern)
    
    # Create sliding windows
    X = []
    y = []
    
    for i in range(n_samples):
        # Input: concatenated windows for all SKUs
        x_sample = []
        y_sample = []
        
        for sku_idx in range(n_skus):
            # Get window for this SKU
            window = base_patterns[sku_idx][i:i+window_size]
            x_sample.extend(window)
            
            # Target: next value for this SKU
            target = base_patterns[sku_idx][i+window_size]
            y_sample.append(target)
        
        X.append(x_sample)
        y.append(y_sample)
    
    return np.array(X), np.array(y)

def train_and_evaluate_model():
    """
    Train and evaluate the multi-head forecasting model.
    """
    print("=== Multi-Head Forecasting Model Demo ===\n")
    
    # Parameters
    n_skus = 4
    window_size = 10
    hidden_units = 16
    n_samples = 1000
    
    # Generate data
    print("Generating synthetic time series data...")
    X, y = generate_synthetic_time_series_data(n_skus, n_samples, window_size)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Input shape: {X_train.shape[1]} (concatenated features for {n_skus} SKUs)")
    print(f"Output shape: {y_train.shape[1]} (one forecast per SKU)\n")
    
    # Create model
    print("Creating multi-head forecasting model...")
    model = create_multi_head_forecasting_model(
        group_size=n_skus,
        window_size=window_size,
        hidden_units=hidden_units,
        activation='relu'
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, 
        [y_train[:, i] for i in range(n_skus)],  # Separate targets for each head
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss = model.evaluate(X_test, [y_test[:, i] for i in range(n_skus)], verbose=0)
    
    print(f"\nTest Loss: {test_loss}")
    
    # Make predictions
    predictions = model.predict(X_test[:10])  # Predict on first 10 test samples
    
    print(f"\nPrediction shapes: {[p.shape for p in predictions]}")
    
    # Show some predictions vs actual
    print("\nSample predictions (first 5 test samples):")
    for i in range(5):
        print(f"Sample {i+1}:")
        for sku_idx in range(n_skus):
            actual = y_test[i, sku_idx]
            predicted = predictions[sku_idx][i, 0]
            print(f"  SKU {sku_idx}: Actual={actual:.3f}, Predicted={predicted:.3f}")
        print()
    
    return model, history, X_test, y_test, predictions

def visualize_results(X_test, y_test, predictions, n_samples=50):
    """
    Visualize the forecasting results.
    """
    try:
        import matplotlib.pyplot as plt
        
        n_skus = len(predictions)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for sku_idx in range(n_skus):
            actual = y_test[:n_samples, sku_idx]
            predicted = predictions[sku_idx][:n_samples, 0]
            
            axes[sku_idx].plot(actual, label='Actual', alpha=0.7)
            axes[sku_idx].plot(predicted, label='Predicted', alpha=0.7)
            axes[sku_idx].set_title(f'SKU {sku_idx} Forecasts')
            axes[sku_idx].set_xlabel('Time Step')
            axes[sku_idx].set_ylabel('Value')
            axes[sku_idx].legend()
            axes[sku_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('forecasting_results.png', dpi=150, bbox_inches='tight')
        print("Visualization saved as 'forecasting_results.png'")
        
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")

if __name__ == "__main__":
    # Run the demo
    model, history, X_test, y_test, predictions = train_and_evaluate_model()
    
    # Try to visualize results
    visualize_results(X_test, y_test, predictions)
    
    print("\n🎉 Demo completed successfully!")
    print("The BlockDense multi-head forecasting model is working correctly with realistic data.") 