import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class BlockDense(layers.Layer):
    """
    A custom Keras layer that implements completely independent SKU processing.
    
    This layer maintains G completely independent "mini-trunks" with NO weight sharing.
    Each SKU has its own separate weight matrix and processes its data independently.
    
    Attributes:
        group_size (int): Number of independent SKUs G
        window_size (int): Input width per SKU w
        hidden_units (int): Output width per SKU H
        activation (str|None): Activation function to apply after matrix multiplication
        sku_weights (list): List of independent weight matrices for each SKU
    """
    
    def __init__(self, group_size, window_size, hidden_units, activation=None, **kwargs):
        """
        Initialize the BlockDense layer.
        
        Args:
            group_size (int): Number of independent SKUs G
            window_size (int): Input width per SKU w
            hidden_units (int): Output width per SKU H
            activation (str|None): Activation function name (e.g., 'relu', 'tanh', None)
            **kwargs: Additional arguments passed to the parent Layer class
        """
        super(BlockDense, self).__init__(**kwargs)
        
        self.group_size = group_size
        self.window_size = window_size
        self.hidden_units = hidden_units
        self.activation = activation
        
        # Validate inputs
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if hidden_units <= 0:
            raise ValueError("hidden_units must be positive")
    
    def build(self, input_shape):
        """
        Build the layer by creating separate weight matrices for each SKU.
        
        Args:
            input_shape: Expected input shape (batch_size, window_size * group_size)
        """
        # Validate input shape
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2D input, got shape {input_shape}")
        
        expected_input_dim = self.window_size * self.group_size
        if input_shape[-1] != expected_input_dim:
            raise ValueError(f"Expected input dimension {expected_input_dim}, got {input_shape[-1]}")
        
        # Create separate weight matrices for each SKU
        self.sku_weights = []
        for i in range(self.group_size):
            weight = self.add_weight(
                name=f'sku_{i}_weights',
                shape=(self.window_size, self.hidden_units),
                initializer=keras.initializers.GlorotUniform(),
                trainable=True
            )
            self.sku_weights.append(weight)
        
        # Set the activation function
        if self.activation is not None:
            self.activation_fn = keras.activations.get(self.activation)
        else:
            self.activation_fn = None
    
    def call(self, inputs, training=None):
        """
        Forward pass of the BlockDense layer.
        
        Args:
            inputs: Input tensor of shape (batch_size, window_size * group_size)
            training: Boolean indicating if in training mode (unused in this implementation)
            
        Returns:
            Output tensor of shape (batch_size, hidden_units * group_size)
        """
        outputs = []
        
        # Process each SKU independently
        for i in range(self.group_size):
            # Extract input slice for this SKU
            start_idx = i * self.window_size
            end_idx = (i + 1) * self.window_size
            sku_input = inputs[:, start_idx:end_idx]
            
            # Apply this SKU's independent weight matrix
            sku_output = tf.matmul(sku_input, self.sku_weights[i])
            
            # Apply activation function if specified
            if self.activation_fn is not None:
                sku_output = self.activation_fn(sku_output)
            
            outputs.append(sku_output)
        
        # Concatenate all SKU outputs
        return tf.concat(outputs, axis=1)
    
    def get_config(self):
        """
        Get the layer configuration for serialization.
        
        Returns:
            Dictionary containing the layer configuration
        """
        config = super(BlockDense, self).get_config()
        config.update({
            'group_size': self.group_size,
            'window_size': self.window_size,
            'hidden_units': self.hidden_units,
            'activation': self.activation,
        })
        return config
    
    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.
        
        Args:
            input_shape: Input shape tuple
            
        Returns:
            Output shape tuple
        """
        return (input_shape[0], self.hidden_units * self.group_size)


def create_multi_head_forecasting_model(group_size, window_size, hidden_units, activation='relu'):
    """
    Create a multi-head forecasting model using BlockDense layer.
    
    Args:
        group_size (int): Number of SKUs (independent blocks)
        window_size (int): Look-back window size per SKU
        hidden_units (int): Hidden units per SKU
        activation (str): Activation function for BlockDense layer
        
    Returns:
        tf.keras.Model: Compiled multi-head forecasting model
    """
    # Input layer
    inp = keras.Input(shape=(window_size * group_size,), name='input')
    
    # BlockDense trunk layer
    x = BlockDense(
        group_size=group_size,
        window_size=window_size,
        hidden_units=hidden_units,
        activation=activation,
        name='block_dense_trunk'
    )(inp)
    
    # Split and attach individual heads for each SKU
    outputs = []
    for i in range(group_size):
        # Extract the slice corresponding to this SKU
        start = i * hidden_units
        end = (i + 1) * hidden_units
        
        # Lambda layer to extract the specific slice
        xi = layers.Lambda(
            lambda t, start_idx=start, end_idx=end: t[:, start_idx:end_idx],
            name=f'sku_slice_{i}'
        )(x)
        
        # Individual head for this SKU
        head = layers.Dense(
            1,
            activation='linear',
            name=f"sku_head_{i}"
        )(xi)
        
        outputs.append(head)
    
    # Create the model
    model = keras.Model(inputs=inp, outputs=outputs, name="grouped_sku_model")
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    return model


if __name__ == "__main__":
    # Testing snippet
    print("Testing BlockDense layer and multi-head forecasting model...")
    
    # Parameters
    G = 4            # e.g. 4 SKUs in this group
    w = 10           # look-back window
    H = 16           # hidden units per SKU
    
    # Build model using the helper function
    model = create_multi_head_forecasting_model(G, w, H, activation='relu')
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Dummy data
    x = tf.random.uniform((2, w*G))
    y = [tf.random.uniform((2,1)) for _ in range(G)]
    
    # Verify shapes
    preds = model(x)
    print(f"\nPrediction shapes: {[p.shape for p in preds]}")  # should be [(2,1), ...] G times
    
    # Test training step
    print("\nTesting training step...")
    loss = model.train_on_batch(x, y)
    print(f"Training loss: {loss}")
    
    print("\nAll tests passed! ✅") 