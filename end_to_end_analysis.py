import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from deep_block_dense_layer import create_deep_multi_head_forecasting_model, create_individual_deep_models
from block_dense_layer import create_multi_head_forecasting_model

def calculate_forecast_metrics(y_true, y_pred):
    """
    Calculate Bias, MAE, MAPE, and WAPE for univariate time series forecasting.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary with Bias, MAE, MAPE, and WAPE metrics
    """
    # Ensure arrays are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Bias (Mean Forecast Error)
    bias = np.mean(y_pred - y_true)
    
    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    # Handle division by zero
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
    
    # WAPE (Weighted Absolute Percentage Error)
    # Handle division by zero
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.where(y_true != 0, y_true, 1)) * 100
    
    return {
        'Bias': bias,
        'MAE': mae,
        'MAPE': mape,
        'WAPE': wape
    }

def generate_monthly_synthetic_data(n_skus=10, years=5, window_size=12):
    """
    Generate realistic 5-year monthly synthetic data for multiple SKUs.
    
    Args:
        n_skus: Number of SKUs
        years: Number of years of data
        window_size: Lookback window size (months)
    
    Returns:
        X: Input features (sliding windows)
        y: Target values (next month)
        sku_data: Original time series data for each SKU
        dates: Date range for plotting
    """
    np.random.seed(42)
    months_per_year = 12
    total_months = years * months_per_year
    
    # Generate date range
    dates = pd.date_range(start='2019-01-01', periods=total_months, freq='M')
    
    sku_data = []
    
    for sku in range(n_skus):
        # Base trend (different for each SKU)
        base_trend = np.linspace(0, 20 + sku * 3, total_months)
        
        # Seasonal pattern (annual cycle)
        seasonal = 8 * np.sin(2 * np.pi * np.arange(total_months) / 12) + \
                   4 * np.sin(2 * np.pi * np.arange(total_months) / 6)  # Semi-annual
        
        # Random walk component
        random_walk = np.cumsum(np.random.normal(0, 0.5, total_months))
        
        # Noise
        noise = np.random.normal(0, 1, total_months)
        
        # Combine components with SKU-specific baseline
        sku_series = base_trend + seasonal + random_walk + noise + (sku * 10)
        
        # Ensure positive values (sales/inventory can't be negative)
        sku_series = np.maximum(sku_series, 5)
        
        sku_data.append(sku_series)
    
    # Create sliding windows
    X, y = [], []
    for i in range(total_months - window_size):
        x_sample = []
        y_sample = []
        
        for sku in range(n_skus):
            # Input: window_size months of data
            x_sample.extend(sku_data[sku][i:i+window_size])
            # Target: next month
            y_sample.append(sku_data[sku][i+window_size])
        
        X.append(x_sample)
        y.append(y_sample)
    
    return np.array(X), np.array(y), sku_data, dates

def evaluate_sku_by_sku(models, X_test, y_test, window_size, n_skus, model_name):
    """Evaluate each SKU individually and return detailed metrics."""
    sku_metrics = []
    
    if len(models) == 1:  # Combined model
        pred = models[0].predict(X_test, verbose=0)
        if isinstance(pred, list):
            pred = np.column_stack(pred)
        
        for sku in range(n_skus):
            y_sku = y_test[:, sku]
            p_sku = pred[:, sku]
            
            metrics = calculate_forecast_metrics(y_sku, p_sku)
            
            sku_metrics.append({
                'SKU': f'SKU_{sku:02d}',
                'Bias': metrics['Bias'],
                'MAE': metrics['MAE'],
                'MAPE': metrics['MAPE'],
                'WAPE': metrics['WAPE']
            })
    else:  # Individual models
        for sku in range(n_skus):
            X_sku_test = X_test[:, sku*window_size:(sku+1)*window_size]
            y_sku = y_test[:, sku]
            
            p_sku = models[sku].predict(X_sku_test, verbose=0).flatten()
            
            metrics = calculate_forecast_metrics(y_sku, p_sku)
            
            sku_metrics.append({
                'SKU': f'SKU_{sku:02d}',
                'Bias': metrics['Bias'],
                'MAE': metrics['MAE'],
                'MAPE': metrics['MAPE'],
                'WAPE': metrics['WAPE']
            })
    
    return sku_metrics

def print_sku_comparison_table(sku_metrics, model_name):
    """Print a formatted table of SKU-by-SKU metrics."""
    print(f"\n{model_name} - SKU-by-SKU Performance:")
    print("-" * 90)
    print(f"{'SKU':<10} {'Bias':<12} {'MAE':<12} {'MAPE(%)':<12} {'WAPE(%)':<12}")
    print("-" * 90)
    
    for metrics in sku_metrics:
        print(f"{metrics['SKU']:<10} {metrics['Bias']:<12.4f} {metrics['MAE']:<12.4f} {metrics['MAPE']:<12.2f} {metrics['WAPE']:<12.2f}")
    
    # Calculate averages
    avg_bias = np.mean([m['Bias'] for m in sku_metrics])
    avg_mae = np.mean([m['MAE'] for m in sku_metrics])
    avg_mape = np.mean([m['MAPE'] for m in sku_metrics])
    avg_wape = np.mean([m['WAPE'] for m in sku_metrics])
    
    print("-" * 90)
    print(f"{'AVG':<10} {avg_bias:<12.4f} {avg_mae:<12.4f} {avg_mape:<12.2f} {avg_wape:<12.2f}")
    print()

def plot_sku_data(sku_data, dates, n_skus):
    """Plot the original time series data for all SKUs."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for sku in range(n_skus):
        ax = axes[sku]
        ax.plot(dates, sku_data[sku], linewidth=2, label=f'SKU_{sku:02d}')
        ax.set_title(f'SKU_{sku:02d} - 5 Years Monthly Data')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('sku_time_series.png', dpi=300, bbox_inches='tight')
    plt.show()

def end_to_end_analysis():
    """Complete end-to-end analysis with 10 SKUs and 5 years of monthly data."""
    
    print("="*80)
    print("END-TO-END ANALYSIS: 10 SKUs with 5 Years Monthly Data")
    print("UNIVARIATE FORECASTING METRICS: Bias, MAE, MAPE, WAPE")
    print("="*80)
    
    # Configuration
    n_skus = 10
    years = 5
    window_size = 12  # 12-month lookback
    epochs = 100
    batch_size = 32
    
    print(f"Configuration:")
    print(f"  - SKUs: {n_skus}")
    print(f"  - Time period: {years} years ({years * 12} months)")
    print(f"  - Window size: {window_size} months")
    print(f"  - Training epochs: {epochs}")
    print(f"  - Evaluation metrics: Bias, MAE, MAPE, WAPE")
    print()
    
    # Generate synthetic data
    print("1. GENERATING SYNTHETIC DATA")
    print("-" * 40)
    X, y, sku_data, dates = generate_monthly_synthetic_data(n_skus, years, window_size)
    
    print(f"Data generated:")
    print(f"  - Total samples: {X.shape[0]}")
    print(f"  - Input features: {X.shape[1]} ({window_size} months × {n_skus} SKUs)")
    print(f"  - Target outputs: {y.shape[1]} ({n_skus} SKUs)")
    print(f"  - Time period: {dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}")
    print()
    
    # Plot original data
    print("2. VISUALIZING ORIGINAL DATA")
    print("-" * 40)
    plot_sku_data(sku_data, dates, n_skus)
    
    # Split data
    train_size = int(0.8 * X.shape[0])
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Split y for individual models
    y_train_split = [y_train[:, i:i+1] for i in range(n_skus)]
    y_test_split = [y_test[:, i:i+1] for i in range(n_skus)]
    
    print(f"Data split:")
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Testing samples: {X_test.shape[0]}")
    print()
    
    results = {}
    
    # Test 1: Shallow models (single layer)
    print("3. SHALLOW MODELS (Single Layer)")
    print("-" * 40)
    
    # Individual shallow models
    print("Training individual shallow models...")
    start_time = time.time()
    individual_shallow = []
    for i in range(n_skus):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(window_size,)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train[:, i*window_size:(i+1)*window_size], y_train_split[i], 
                 epochs=epochs, batch_size=batch_size, verbose=0)
        individual_shallow.append(model)
    individual_shallow_time = time.time() - start_time
    
    # Combined shallow model (BlockDense)
    print("Training combined shallow model (BlockDense)...")
    start_time = time.time()
    combined_shallow = create_multi_head_forecasting_model(n_skus, window_size, 32)
    combined_shallow.fit(X_train, y_train_split, epochs=epochs, batch_size=batch_size, verbose=0)
    combined_shallow_time = time.time() - start_time
    
    # Test 2: Deep models with single activation
    print("\n4. DEEP MODELS (Single Activation)")
    print("-" * 40)
    
    hidden_layers = [64, 32, 16]
    
    # Individual deep models
    print("Training individual deep models...")
    start_time = time.time()
    individual_deep = create_individual_deep_models(n_skus, window_size, hidden_layers, activation='relu')
    for i, model in enumerate(individual_deep):
        model.fit(X_train[:, i*window_size:(i+1)*window_size], y_train_split[i], 
                 epochs=epochs, batch_size=batch_size, verbose=0)
    individual_deep_time = time.time() - start_time
    
    # Combined deep model
    print("Training combined deep model (DeepBlockDense)...")
    start_time = time.time()
    combined_deep = create_deep_multi_head_forecasting_model(n_skus, window_size, hidden_layers, activation='relu')
    combined_deep.fit(X_train, y_train_split, epochs=epochs, batch_size=batch_size, verbose=0)
    combined_deep_time = time.time() - start_time
    
    # Test 3: Deep models with multiple activations
    print("\n5. DEEP MODELS (Multiple Activations)")
    print("-" * 40)
    
    activations = ['relu', 'tanh', 'sigmoid']
    
    # Individual deep models with multiple activations
    print("Training individual deep models with multiple activations...")
    start_time = time.time()
    individual_deep_multi = create_individual_deep_models(n_skus, window_size, hidden_layers, activation=activations)
    for i, model in enumerate(individual_deep_multi):
        model.fit(X_train[:, i*window_size:(i+1)*window_size], y_train_split[i], 
                 epochs=epochs, batch_size=batch_size, verbose=0)
    individual_deep_multi_time = time.time() - start_time
    
    # Combined deep model with multiple activations
    print("Training combined deep model with multiple activations...")
    start_time = time.time()
    combined_deep_multi = create_deep_multi_head_forecasting_model(n_skus, window_size, hidden_layers, activation=activations)
    combined_deep_multi.fit(X_train, y_train_split, epochs=epochs, batch_size=batch_size, verbose=0)
    combined_deep_multi_time = time.time() - start_time
    
    # Evaluate all models with SKU-by-SKU comparison
    print("\n6. SKU-BY-SKU EVALUATION")
    print("="*80)
    
    models_to_evaluate = [
        ("Individual Shallow", individual_shallow, individual_shallow_time),
        ("Combined Shallow", [combined_shallow], combined_shallow_time),
        ("Individual Deep", individual_deep, individual_deep_time),
        ("Combined Deep", [combined_deep], combined_deep_time),
        ("Individual Deep Multi", individual_deep_multi, individual_deep_multi_time),
        ("Combined Deep Multi", [combined_deep_multi], combined_deep_multi_time)
    ]
    
    all_sku_metrics = {}
    
    for name, models, train_time in models_to_evaluate:
        print(f"\n{name}:")
        print(f"  Training time: {train_time:.2f}s")
        
        # Get SKU-by-SKU metrics
        sku_metrics = evaluate_sku_by_sku(models, X_test, y_test, window_size, n_skus, name)
        all_sku_metrics[name] = sku_metrics
        
        # Print SKU comparison table
        print_sku_comparison_table(sku_metrics, name)
        
        # Overall metrics
        avg_bias = np.mean([m['Bias'] for m in sku_metrics])
        avg_mae = np.mean([m['MAE'] for m in sku_metrics])
        avg_mape = np.mean([m['MAPE'] for m in sku_metrics])
        avg_wape = np.mean([m['WAPE'] for m in sku_metrics])
        
        print(f"  Overall - Bias: {avg_bias:.4f}, MAE: {avg_mae:.4f}, MAPE: {avg_mape:.2f}%, WAPE: {avg_wape:.2f}%")
        
        results[name] = {
            'train_time': train_time,
            'avg_bias': avg_bias,
            'avg_mae': avg_mae,
            'avg_mape': avg_mape,
            'avg_wape': avg_wape,
            'sku_metrics': sku_metrics
        }
    
    # Create comprehensive comparison table
    print("\n7. COMPREHENSIVE COMPARISON TABLE")
    print("="*110)
    
    print(f"{'Model':<25} {'Training Time':<15} {'Avg Bias':<12} {'Avg MAE':<12} {'Avg MAPE(%)':<12} {'Avg WAPE(%)':<12}")
    print("-" * 110)
    
    for name in results.keys():
        result = results[name]
        print(f"{name:<25} {result['train_time']:<15.2f}s {result['avg_bias']:<12.4f} {result['avg_mae']:<12.4f} {result['avg_mape']:<12.2f} {result['avg_wape']:<12.2f}")
    
    # Create SKU-by-SKU comparison table
    print("\n8. SKU-BY-SKU COMPARISON TABLE")
    print("="*140)
    
    # Header
    header = f"{'SKU':<10}"
    for name in results.keys():
        header += f"{name[:15]:<15}"
    print(header)
    print("-" * 140)
    
    # MAE comparison
    print("MAE Comparison:")
    for sku_idx in range(n_skus):
        row = f"SKU_{sku_idx:02d}  "
        for name in results.keys():
            mae = results[name]['sku_metrics'][sku_idx]['MAE']
            row += f"{mae:<15.4f}"
        print(row)
    
    print()
    print("MAPE (%) Comparison:")
    for sku_idx in range(n_skus):
        row = f"SKU_{sku_idx:02d}  "
        for name in results.keys():
            mape = results[name]['sku_metrics'][sku_idx]['MAPE']
            row += f"{mape:<15.2f}"
        print(row)
    
    print()
    print("WAPE (%) Comparison:")
    for sku_idx in range(n_skus):
        row = f"SKU_{sku_idx:02d}  "
        for name in results.keys():
            wape = results[name]['sku_metrics'][sku_idx]['WAPE']
            row += f"{wape:<15.2f}"
        print(row)
    
    # Create comparison plots
    print("\n9. GENERATING COMPARISON PLOTS")
    print("-" * 40)
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 24))
    
    # Training time comparison
    names = list(results.keys())
    train_times = [results[name]['train_time'] for name in names]
    
    axes[0, 0].bar(names, train_times, color=['red', 'blue', 'red', 'blue', 'red', 'blue'])
    axes[0, 0].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # MAE comparison
    mae_values = [results[name]['avg_mae'] for name in names]
    axes[0, 1].bar(names, mae_values, color=['red', 'blue', 'red', 'blue', 'red', 'blue'])
    axes[0, 1].set_title('Average MAE Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # MAPE comparison
    mape_values = [results[name]['avg_mape'] for name in names]
    axes[1, 0].bar(names, mape_values, color=['red', 'blue', 'red', 'blue', 'red', 'blue'])
    axes[1, 0].set_title('Average MAPE Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('MAPE (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # WAPE comparison
    wape_values = [results[name]['avg_wape'] for name in names]
    axes[1, 1].bar(names, wape_values, color=['red', 'blue', 'red', 'blue', 'red', 'blue'])
    axes[1, 1].set_title('Average WAPE Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('WAPE (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # SKU-by-SKU MAE heatmap
    sku_mae_data = []
    for name in results.keys():
        sku_mae_data.append([results[name]['sku_metrics'][sku]['MAE'] for sku in range(n_skus)])
    
    im = axes[2, 0].imshow(sku_mae_data, cmap='viridis', aspect='auto')
    axes[2, 0].set_title('SKU-by-SKU MAE Heatmap', fontsize=14, fontweight='bold')
    axes[2, 0].set_xlabel('SKU Index')
    axes[2, 0].set_ylabel('Model')
    axes[2, 0].set_yticks(range(len(names)))
    axes[2, 0].set_yticklabels([name[:15] for name in names])
    axes[2, 0].set_xticks(range(n_skus))
    axes[2, 0].set_xticklabels([f'SKU_{i:02d}' for i in range(n_skus)])
    plt.colorbar(im, ax=axes[2, 0])
    
    # SKU-by-SKU MAPE heatmap
    sku_mape_data = []
    for name in results.keys():
        sku_mape_data.append([results[name]['sku_metrics'][sku]['MAPE'] for sku in range(n_skus)])
    
    im = axes[2, 1].imshow(sku_mape_data, cmap='plasma', aspect='auto')
    axes[2, 1].set_title('SKU-by-SKU MAPE Heatmap', fontsize=14, fontweight='bold')
    axes[2, 1].set_xlabel('SKU Index')
    axes[2, 1].set_ylabel('Model')
    axes[2, 1].set_yticks(range(len(names)))
    axes[2, 1].set_yticklabels([name[:15] for name in names])
    axes[2, 1].set_xticks(range(n_skus))
    axes[2, 1].set_xticklabels([f'SKU_{i:02d}' for i in range(n_skus)])
    plt.colorbar(im, ax=axes[2, 1])
    
    plt.tight_layout()
    plt.savefig('end_to_end_benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nTraining Time Speedup:")
    for i in range(0, len(names), 2):
        individual_time = results[names[i]]['train_time']
        combined_time = results[names[i+1]]['train_time']
        speedup = individual_time / combined_time
        print(f"  {names[i]} vs {names[i+1]}: {speedup:.2f}x faster")
    
    print("\nPerformance Comparison:")
    for i in range(0, len(names), 2):
        individual_mae = results[names[i]]['avg_mae']
        combined_mae = results[names[i+1]]['avg_mae']
        individual_mape = results[names[i]]['avg_mape']
        combined_mape = results[names[i+1]]['avg_mape']
        
        print(f"  {names[i]}: MAE={individual_mae:.4f}, MAPE={individual_mape:.2f}%")
        print(f"  {names[i+1]}: MAE={combined_mae:.4f}, MAPE={combined_mape:.2f}%")
        
        # Show best performing SKU for each model (lowest MAPE)
        best_sku_individual = min(results[names[i]]['sku_metrics'], key=lambda x: x['MAPE'])
        best_sku_combined = min(results[names[i+1]]['sku_metrics'], key=lambda x: x['MAPE'])
        
        print(f"    Best SKU - Individual: {best_sku_individual['SKU']} (MAPE={best_sku_individual['MAPE']:.2f}%)")
        print(f"    Best SKU - Combined: {best_sku_combined['SKU']} (MAPE={best_sku_combined['MAPE']:.2f}%)")
        print()
    
    print("✅ End-to-end analysis completed!")
    print("📊 Check 'sku_time_series.png' for original data visualization")
    print("📊 Check 'end_to_end_benchmark_results.png' for comparison results")

if __name__ == "__main__":
    end_to_end_analysis() 