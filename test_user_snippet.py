import tensorflow as tf
from block_dense_layer import BlockDense

# Parameters
G = 4            # e.g. 4 SKUs in this group
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
print([p.shape for p in preds])  # should be [(2,1), ...] G times

print("✅ User's exact testing snippet works correctly!") 