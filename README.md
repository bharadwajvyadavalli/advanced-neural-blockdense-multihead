# Advanced Neural BlockDense Multi-Head Forecasting

A TensorFlow/Keras implementation of **completely independent SKU forecasting** using custom BlockDense layers with **zero weight sharing** between SKUs.

## 🎯 Key Features

- **🔒 Complete SKU Independence**: Each SKU has its own separate weight matrices with **NO weight sharing whatsoever**
- **🏗️ Deep Architectures**: Support for multiple hidden layers while maintaining SKU independence
- **🎛️ Multiple Activation Functions**: Different activation functions for each layer (relu, tanh, sigmoid, etc.)
- **📊 Univariate Forecasting Metrics**: Bias, MAE, MAPE, WAPE for proper time series evaluation
- **⚡ Efficient Training**: Combined models for faster training while maintaining SKU independence
- **📈 Comprehensive Benchmarking**: Compare individual vs. combined approaches

## 🏗️ Architecture

### BlockDense Layer
- **Independent Weight Matrices**: Each SKU has its own separate `(window_size × hidden_units)` weight matrix
- **No Cross-SKU Connections**: SKUs cannot influence each other's weights during training
- **Parallel Processing**: All SKUs processed simultaneously for efficiency
- **Separate Processing**: Each SKU's input is processed independently with its own weights

### DeepBlockDense Layer
- **Multi-Layer Independence**: Each SKU has separate weight matrices for each layer
- **Layer-by-Layer Processing**: Independent forward pass through all layers for each SKU
- **Multiple Activations**: Support for different activation functions per layer

## 🔬 Technical Implementation

### Weight Independence
Each SKU has completely separate weight matrices:

```
SKU 0: W^(0) ∈ ℝ^(w × H)  (independent weight matrix)
SKU 1: W^(1) ∈ ℝ^(w × H)  (independent weight matrix)
SKU 2: W^(2) ∈ ℝ^(w × H)  (independent weight matrix)
...
SKU G: W^(G) ∈ ℝ^(w × H)  (independent weight matrix)
```

### Processing Flow
```
Input: [x^(0), x^(1), ..., x^(G)] ∈ ℝ^(wG)
For each SKU i:
  y^(i) = W^(i) × x^(i) ∈ ℝ^H
Output: [y^(0), y^(1), ..., y^(G)] ∈ ℝ^(HG)
```

### Deep Architecture
For deep networks, each SKU has separate weight matrices for each layer:
```
SKU i, Layer j: W^(i,j) ∈ ℝ^(H_{j-1} × H_j)
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run comprehensive analysis
python3 end_to_end_analysis.py
```

## 📊 Analysis Results

The analysis provides:
- **SKU-by-SKU Performance**: Individual metrics for each SKU
- **Training Time Comparison**: Speedup benefits of combined models
- **Forecasting Accuracy**: Bias, MAE, MAPE, WAPE metrics
- **Visual Comparisons**: Heatmaps and bar charts

## 🔧 Key Benefits

1. **True SKU Independence**: No weight sharing ensures each SKU learns independently
2. **Computational Efficiency**: Combined models train faster than individual models
3. **Flexible Architecture**: Support for shallow and deep networks
4. **Multiple Activations**: Different activation functions for different layers
5. **Proper Metrics**: Univariate time series forecasting metrics

## 📁 Project Structure

```
├── block_dense_layer.py          # Core BlockDense implementation
├── deep_block_dense_layer.py     # Deep BlockDense with multiple activations
├── end_to_end_analysis.py        # Complete analysis script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🎯 Use Cases

- **Multi-SKU Forecasting**: Independent forecasting for multiple products
- **Supply Chain Optimization**: SKU-specific demand prediction
- **Inventory Management**: Independent stock level forecasting
- **Retail Analytics**: Product-specific sales prediction

## 🔬 Technical Details

### Weight Independence
- Each SKU has completely separate weight matrices
- No gradients flow between different SKUs
- Independent learning for each SKU's patterns
- No shared parameters or cross-SKU influence

### Model Variants
1. **Individual Models**: Separate neural network for each SKU
2. **Combined Shallow**: Single BlockDense layer for all SKUs
3. **Combined Deep**: Multiple BlockDense layers for all SKUs
4. **Multi-Activation**: Different activation functions per layer

### Evaluation Metrics
- **Bias**: Systematic over/under-prediction
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **WAPE**: Weighted Absolute Percentage Error

## 🏗️ Architecture Comparison

### Individual vs Combined
- **Individual**: Separate neural networks, independent training
- **Combined**: Single network with independent weight matrices per SKU
- **Performance**: Identical accuracy, combined is faster
- **Memory**: Combined uses less memory

### Shallow vs Deep
- **Shallow**: Single hidden layer, fewer parameters
- **Deep**: Multiple hidden layers, more parameters
- **Data Requirements**: Deep needs more data to avoid overfitting
- **Complexity**: Deep can capture more complex patterns

### Single vs Multi-Activation
- **Single**: Same activation function for all layers
- **Multi**: Different activation functions per layer
- **Stability**: Single activation is more stable with limited data
- **Complexity**: Multi-activation adds complexity without benefits for small datasets