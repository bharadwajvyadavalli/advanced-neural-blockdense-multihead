import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class DeepBlockDense(layers.Layer):
    """
    A deep BlockDense layer that supports multiple hidden layers while maintaining
    complete SKU independence with NO weight sharing.
    
    This layer creates a deep network where each SKU has its own completely independent
    "mini-trunk" through multiple layers, with separate weight matrices for each SKU.
    
    Attributes:
        group_size (int): Number of independent SKUs G
        window_size (int): Input width per SKU w
        hidden_layers (list): List of hidden units for each layer
        activation (str|None): Activation function to apply after each layer
        dropout_rate (float): Dropout rate for regularization
        sku_layers (list): List of independent layer weights for each SKU
        dropout_layers (list): List of dropout layers
    """
    
    def __init__(self, group_size, window_size, hidden_layers, activation='relu', 
                 dropout_rate=0.1, **kwargs):
        """
        Initialize the DeepBlockDense layer.
        
        Args:
            group_size (int): Number of independent SKUs G
            window_size (int): Input width per SKU w
            hidden_layers (list): List of hidden units for each layer [H1, H2, H3, ...]
            activation (str|list): Activation function(s). Can be:
                - str: Single activation for all layers (e.g., 'relu', 'tanh')
                - list: Different activation for each layer (e.g., ['relu', 'tanh', 'sigmoid'])
            dropout_rate (float): Dropout rate for regularization
            **kwargs: Additional arguments passed to the parent Layer class
        """
        super(DeepBlockDense, self).__init__(**kwargs)
        
        self.group_size = group_size
        self.window_size = window_size
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        
        # Handle activation functions
        if isinstance(activation, str):
            # Single activation for all layers
            self.activations = [activation] * len(hidden_layers)
        elif isinstance(activation, list):
            # Different activation for each layer
            if len(activation) != len(hidden_layers):
                raise ValueError(f"Number of activations ({len(activation)}) must match number of layers ({len(hidden_layers)})")
            self.activations = activation
        else:
            raise ValueError("activation must be a string or list of strings")
        
        # Validate inputs
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not hidden_layers or not all(h > 0 for h in hidden_layers):
            raise ValueError("hidden_layers must be a non-empty list of positive integers")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
    
    def build(self, input_shape):
        """
        Build the deep layer by creating separate weight matrices for each SKU and layer.
        
        Args:
            input_shape: Expected input shape (batch_size, window_size * group_size)
        """
        # Validate input shape
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2D input, got shape {input_shape}")
        
        expected_input_dim = self.window_size * self.group_size
        if input_shape[-1] != expected_input_dim:
            raise ValueError(f"Expected input dimension {expected_input_dim}, got {input_shape[-1]}")
        
        # Create independent weight matrices for each SKU and layer
        self.sku_layers = []
        
        for sku_idx in range(self.group_size):
            sku_layer_weights = []
            current_input_dim = self.window_size
            
            for layer_idx, (hidden_units, activation) in enumerate(zip(self.hidden_layers, self.activations)):
                # Create weight matrix for this SKU and layer
                weight = self.add_weight(
                    name=f'sku_{sku_idx}_layer_{layer_idx}_weights',
                    shape=(current_input_dim, hidden_units),
                    initializer=keras.initializers.GlorotUniform(),
                    trainable=True
                )
                sku_layer_weights.append(weight)
                current_input_dim = hidden_units
            
            self.sku_layers.append(sku_layer_weights)
        
        # Create dropout layers (shared across SKUs for efficiency, but applied independently)
        self.dropout_layers = []
        for i in range(len(self.hidden_layers) - 1):
            dropout = layers.Dropout(
                rate=self.dropout_rate,
                name=f'dropout_layer_{i}'
            )
            self.dropout_layers.append(dropout)
    
    def call(self, inputs, training=None):
        """
        Forward pass through the deep BlockDense network.
        
        Args:
            inputs: Input tensor of shape (batch_size, window_size * group_size)
            training: Boolean indicating if in training mode
            
        Returns:
            Output tensor of shape (batch_size, hidden_layers[-1] * group_size)
        """
        sku_outputs = []
        
        # Process each SKU independently through all layers
        for sku_idx in range(self.group_size):
            # Extract input slice for this SKU
            start_idx = sku_idx * self.window_size
            end_idx = (sku_idx + 1) * self.window_size
            sku_input = inputs[:, start_idx:end_idx]
            
            # Pass through all layers for this SKU
            x = sku_input
            for layer_idx, (weight, activation) in enumerate(zip(self.sku_layers[sku_idx], self.activations)):
                # Apply weight matrix
                x = tf.matmul(x, weight)
                
                # Apply activation function
                if activation is not None:
                    activation_fn = keras.activations.get(activation)
                    x = activation_fn(x)
                
                # Apply dropout (except for the last layer)
                if layer_idx < len(self.sku_layers[sku_idx]) - 1 and training:
                    x = self.dropout_layers[layer_idx](x)
            
            sku_outputs.append(x)
        
        # Concatenate all SKU outputs
        return tf.concat(sku_outputs, axis=1)
    
    def get_config(self):
        """
        Get the layer configuration for serialization.
        
        Returns:
            Dictionary containing the layer configuration
        """
        config = super(DeepBlockDense, self).get_config()
        config.update({
            'group_size': self.group_size,
            'window_size': self.window_size,
            'hidden_layers': self.hidden_layers,
            'activation': self.activations,  # Now stores the list of activations
            'dropout_rate': self.dropout_rate,
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
        return (input_shape[0], self.hidden_layers[-1] * self.group_size)


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


def create_deep_multi_head_forecasting_model(group_size, window_size, hidden_layers, 
                                           activation='relu', dropout_rate=0.1):
    """
    Create a deep multi-head forecasting model using DeepBlockDense layer.
    
    Args:
        group_size (int): Number of SKUs (independent blocks)
        window_size (int): Look-back window size per SKU
        hidden_layers (list): List of hidden units for each layer [H1, H2, H3, ...]
        activation (str|list): Activation function(s). Can be:
            - str: Single activation for all layers (e.g., 'relu', 'tanh')
            - list: Different activation for each layer (e.g., ['relu', 'tanh', 'sigmoid'])
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        tf.keras.Model: Compiled deep multi-head forecasting model
    """
    # Input layer
    inp = keras.Input(shape=(window_size * group_size,), name='input')
    
    # Deep BlockDense trunk layers
    x = DeepBlockDense(
        group_size=group_size,
        window_size=window_size,
        hidden_layers=hidden_layers,
        activation=activation,
        dropout_rate=dropout_rate,
        name='deep_block_dense_trunk'
    )(inp)
    
    # Split and attach individual heads for each SKU
    outputs = []
    final_hidden_units = hidden_layers[-1]
    
    for i in range(group_size):
        # Extract the slice corresponding to this SKU
        start = i * final_hidden_units
        end = (i + 1) * final_hidden_units
        
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
    model = keras.Model(inputs=inp, outputs=outputs, name="deep_grouped_sku_model")
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    return model


def create_individual_deep_models(n_skus, window_size, hidden_layers, activation='relu', dropout_rate=0.1):
    """
    Create individual deep neural networks for each SKU.
    
    Args:
        n_skus (int): Number of SKUs
        window_size (int): Look-back window size
        hidden_layers (list): List of hidden units for each layer
        activation (str|list): Activation function(s). Can be:
            - str: Single activation for all layers (e.g., 'relu', 'tanh')
            - list: Different activation for each layer (e.g., ['relu', 'tanh', 'sigmoid'])
        dropout_rate (float): Dropout rate
        
    Returns:
        list: List of individual deep models
    """
    models = []
    
    # Handle activation functions for individual models
    if isinstance(activation, str):
        # Single activation for all layers
        activations = [activation] * len(hidden_layers)
    elif isinstance(activation, list):
        # Different activation for each layer
        if len(activation) != len(hidden_layers):
            raise ValueError(f"Number of activations ({len(activation)}) must match number of layers ({len(hidden_layers)})")
        activations = activation
    else:
        raise ValueError("activation must be a string or list of strings")
    
    for i in range(n_skus):
        # Input layer for single SKU
        inp = tf.keras.Input(shape=(window_size,), name=f'input_sku_{i}')
        
        # Deep layers
        x = inp
        for j, (hidden_units, layer_activation) in enumerate(zip(hidden_layers, activations)):
            x = tf.keras.layers.Dense(
                hidden_units, 
                activation=layer_activation, 
                name=f'dense{j+1}_sku_{i}'
            )(x)
            
            # Add dropout (except for the last layer)
            if j < len(hidden_layers) - 1:
                x = tf.keras.layers.Dropout(dropout_rate, name=f'dropout{j+1}_sku_{i}')(x)
        
        # Output layer
        output = tf.keras.layers.Dense(1, activation='linear', name=f'output_sku_{i}')(x)
        
        # Create model
        model = tf.keras.Model(inputs=inp, outputs=output, name=f'individual_deep_sku_{i}')
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        models.append(model)
    
    return models


if __name__ == "__main__":
    # Testing snippet for deep BlockDense
    print("Testing Deep BlockDense layer and deep multi-head forecasting model...")
    
    # Parameters
    G = 4            # 4 SKUs in this group
    w = 10           # look-back window
    H_layers = [32, 16, 8]  # Deep architecture: 3 layers
    
    print("1. Testing with single activation function...")
    # Build deep model with single activation
    deep_model = create_deep_multi_head_forecasting_model(G, w, H_layers, activation='relu')
    
    # Print model summary
    print("\nDeep Model Summary (single activation):")
    deep_model.summary()
    
    # Dummy data
    x = tf.random.uniform((2, w*G))
    y = [tf.random.uniform((2,1)) for _ in range(G)]
    
    # Verify shapes
    preds = deep_model(x)
    print(f"\nPrediction shapes: {[p.shape for p in preds]}")  # should be [(2,1), ...] G times
    
    # Test training step
    print("\nTesting training step...")
    loss = deep_model.train_on_batch(x, y)
    print(f"Training loss: {loss}")
    
    print("\n2. Testing with multiple activation functions...")
    # Build deep model with different activations for each layer
    activations = ['relu', 'tanh', 'sigmoid']  # Different activation for each layer
    deep_model_multi = create_deep_multi_head_forecasting_model(G, w, H_layers, activation=activations)
    
    print(f"\nDeep Model Summary (multiple activations: {activations}):")
    deep_model_multi.summary()
    
    # Test training step
    loss_multi = deep_model_multi.train_on_batch(x, y)
    print(f"Training loss (multi-activation): {loss_multi}")
    
    # Test individual deep models with multiple activations
    print("\n3. Testing individual deep models with multiple activations...")
    individual_deep_models = create_individual_deep_models(G, w, H_layers, activation=activations)
    
    for i, model in enumerate(individual_deep_models):
        print(f"Individual deep model {i} summary:")
        model.summary()
        break  # Just show first model for brevity
    
    print("\nAll deep BlockDense tests passed! ✅")
    print("✅ Single activation function works")
    print("✅ Multiple activation functions work")
    print("✅ Individual models with multiple activations work") 