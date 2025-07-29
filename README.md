# BlockDense Multi-Head Forecasting Model

A TensorFlow/Keras implementation of a custom `BlockDense` layer that implements a block-diagonal Dense transform, enabling independent "mini-trunks" for multi-SKU forecasting with separate output heads.

## Overview

This implementation provides:

1. **BlockDense Layer**: A custom Keras layer that maintains G independent "mini-trunks" of size (w → H) fused into one large matrix multiplication with block-diagonal constraints
2. **Multi-Head Forecasting Model**: A complete model architecture that uses BlockDense as a trunk layer with separate output heads for each SKU

## Architecture

### BlockDense Layer

The `BlockDense` layer enforces a block-diagonal structure where:
- Input shape: `(batch_size, window_size * group_size)`
- Output shape: `(batch_size, hidden_units * group_size)`
- Each input block `[i*w:(i+1)*w]` only connects to its own output block `[i*H:(i+1)*H]`

This ensures that each SKU's features are processed independently while sharing the same computational graph.

### Multi-Head Model

The complete model architecture:
1. **Input Layer**: Accepts concatenated time series data for all SKUs
2. **BlockDense Trunk**: Processes each SKU independently with block-diagonal constraints
3. **Output Heads**: Separate Dense(1) layers for each SKU to produce independent forecasts

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic BlockDense Layer Usage

```python
import tensorflow as tf
from block_dense_layer import BlockDense

# Create BlockDense layer
block_dense = BlockDense(
    group_size=4,      # 4 SKUs
    window_size=10,    # 10 time steps lookback
    hidden_units=16,   # 16 hidden units per SKU
    activation='relu'
)

# Input: (batch_size, 40) - concatenated features for 4 SKUs
input_tensor = tf.random.uniform((32, 40))
output = block_dense(input_tensor)  # Shape: (32, 64)
```

### Complete Multi-Head Model

```python
from block_dense_layer import create_multi_head_forecasting_model

# Create the complete model
model = create_multi_head_forecasting_model(
    group_size=4,
    window_size=10,
    hidden_units=16,
    activation='relu'
)

# Input: (batch_size, 40)
# Output: List of 4 tensors, each of shape (batch_size, 1)
```

### Manual Model Construction

```python
import tensorflow as tf
from block_dense_layer import BlockDense

# Parameters
G = 4            # 4 SKUs
w = 10           # look-back window
H = 16           # hidden units per SKU

# Build model
inp = tf.keras.Input(shape=(w*G,))
x = BlockDense(group_size=G, window_size=w, hidden_units=H, activation='relu')(inp)

# Split and attach heads
outputs = []
for i in range(G):
    start = i * H
    end = (i+1) * H
    xi = tf.keras.layers.Lambda(lambda t: t[:, start:end])(x)
    head = tf.keras.layers.Dense(1, activation='linear', name=f"sku_head_{i}")(xi)
    outputs.append(head)

model = tf.keras.Model(inputs=inp, outputs=outputs, name="grouped_sku_model")
model.compile(optimizer='adam', loss='mse')
```

## Testing

Run the comprehensive test suite:

```bash
python example_usage.py
```

This will test:
- BlockDense layer functionality
- Multi-head model construction
- Block-diagonal constraint enforcement
- Gradient flow through masked weights
- Training step execution

## Key Features

### Block-Diagonal Constraint

The layer maintains a block-diagonal mask that ensures:
- Each input block only connects to its corresponding output block
- No cross-talk between different SKUs
- Efficient computation with a single large matrix multiplication

### Gradient Flow

The mask is applied during both forward and backward passes:
- Forward pass: `W_masked = W * mask`
- Backward pass: Gradients are automatically masked through the multiplication

### Activation Support

Supports any activation function from `tf.keras.activations`:
- `'relu'`, `'tanh'`, `'sigmoid'`, etc.
- `None` for linear activation

### Model Serialization

The layer is fully serializable and can be saved/loaded with Keras models.

## File Structure

```
├── block_dense_layer.py    # Main implementation
├── example_usage.py        # Comprehensive test suite
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Requirements

- TensorFlow >= 2.8.0
- NumPy >= 1.21.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.