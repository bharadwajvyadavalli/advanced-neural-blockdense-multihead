import tensorflow as tf
from block_dense_layer import BlockDense, create_multi_head_forecasting_model


def test_block_dense_layer():
    """
    Test the BlockDense layer in isolation.
    """
    print("=== Testing BlockDense Layer ===")
    
    # Parameters
    G = 4            # 4 SKUs in this group
    w = 10           # look-back window
    H = 16           # hidden units per SKU
    
    # Create the layer
    block_dense = BlockDense(
        group_size=G,
        window_size=w,
        hidden_units=H,
        activation='relu'
    )
    
    # Test input
    test_input = tf.random.uniform((2, w*G))
    print(f"Input shape: {test_input.shape}")
    
    # Forward pass
    output = block_dense(test_input)
    print(f"Output shape: {output.shape}")
    
    # Verify the output shape
    expected_output_shape = (2, H*G)
    assert output.shape == expected_output_shape, f"Expected {expected_output_shape}, got {output.shape}"
    print("✅ BlockDense layer test passed!")


def test_multi_head_model():
    """
    Test the complete multi-head forecasting model.
    """
    print("\n=== Testing Multi-Head Forecasting Model ===")
    
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
    
    # Verify all predictions have the correct shape
    for i, pred in enumerate(preds):
        assert pred.shape == (2, 1), f"Prediction {i} has wrong shape: {pred.shape}"
    
    # Test training step
    print("\nTesting training step...")
    loss = model.train_on_batch(x, y)
    print(f"Training loss: {loss}")
    
    print("✅ Multi-head model test passed!")


def test_manual_model_construction():
    """
    Test the exact manual model construction as provided in the requirements.
    """
    print("\n=== Testing Manual Model Construction ===")
    
    # Parameters
    G = 4            # e.g. 4 SKUs in this group
    w = 10           # look-back window
    H = 16           # hidden units per SKU
    
    # Build model manually (exactly as in requirements)
    inp = tf.keras.Input(shape=(w*G,))
    x = BlockDense(group_size=G, window_size=w, hidden_units=H, activation='relu')(inp)
    
    # Split and attach heads
    outputs = []
    for i in range(G):
        start = i * H
        end = (i+1) * H
        xi = tf.keras.layers.Lambda(lambda t: t[:, start:end])(x)
        head = tf.keras.layers.Dense(1,
                     activation='linear',
                     name=f"sku_head_{i}")(xi)
        outputs.append(head)
    
    model = tf.keras.Model(inputs=inp, outputs=outputs, name="grouped_sku_model")
    model.compile(optimizer='adam', loss='mse')
    
    # Dummy data
    x = tf.random.uniform((2, w*G))
    y = [tf.random.uniform((2,1)) for _ in range(G)]
    
    # Verify shapes
    preds = model(x)
    print(f"Prediction shapes: {[p.shape for p in preds]}")  # should be [(2,1), ...] G times
    
    # Verify all predictions have the correct shape
    for i, pred in enumerate(preds):
        assert pred.shape == (2, 1), f"Prediction {i} has wrong shape: {pred.shape}"
    
    print("✅ Manual model construction test passed!")


def test_block_diagonal_constraint():
    """
    Test that the block-diagonal constraint is properly enforced.
    """
    print("\n=== Testing Block-Diagonal Constraint ===")
    
    # Parameters
    G = 3
    w = 5
    H = 4
    
    # Create layer
    block_dense = BlockDense(
        group_size=G,
        window_size=w,
        hidden_units=H,
        activation=None  # No activation for easier testing
    )
    
    # Create test input with only one block active
    test_input = tf.zeros((1, w*G))
    test_input = tf.tensor_scatter_nd_update(
        test_input,
        [[0, 0]],  # Only activate first input
        [1.0]
    )
    
    # Forward pass
    output = block_dense(test_input)
    
    # Check that only the first output block is affected
    # The first H outputs should be non-zero, the rest should be zero
    first_block = output[0, :H]
    other_blocks = output[0, H:]
    
    print(f"First block (should be non-zero): {first_block.numpy()}")
    print(f"Other blocks (should be zero): {other_blocks.numpy()}")
    
    # Verify that other blocks are indeed zero
    assert tf.reduce_all(tf.equal(other_blocks, 0.0)), "Block-diagonal constraint violated!"
    print("✅ Block-diagonal constraint test passed!")


def test_gradient_flow():
    """
    Test that gradients flow properly through the masked weights.
    """
    print("\n=== Testing Gradient Flow ===")
    
    # Parameters
    G = 2
    w = 3
    H = 2
    
    # Create layer
    block_dense = BlockDense(
        group_size=G,
        window_size=w,
        hidden_units=H,
        activation=None
    )
    
    # Create test input
    test_input = tf.random.uniform((1, w*G))
    
    # Compute gradients
    with tf.GradientTape() as tape:
        output = block_dense(test_input)
        loss = tf.reduce_sum(output)
    
    gradients = tape.gradient(loss, block_dense.trainable_variables)
    
    # Check that gradients exist and have the right shape
    assert len(gradients) == 1, "Should have one gradient for the weight matrix"
    assert gradients[0].shape == block_dense.W.shape, "Gradient shape mismatch"
    
    print(f"Gradient shape: {gradients[0].shape}")
    print(f"Weight shape: {block_dense.W.shape}")
    print("✅ Gradient flow test passed!")


if __name__ == "__main__":
    # Run all tests
    test_block_dense_layer()
    test_multi_head_model()
    test_manual_model_construction()
    test_block_diagonal_constraint()
    test_gradient_flow()
    
    print("\n🎉 All tests passed successfully!")
    print("\nThe BlockDense layer and multi-head forecasting model are working correctly.")
    print("You can now use these components in your forecasting pipeline.") 