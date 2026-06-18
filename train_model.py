import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import os

# Load preprocessed data
save_path = 'processed_data'
X_train = np.load(os.path.join(save_path, 'train_X.npy'))
y_train = np.load(os.path.join(save_path, 'train_y.npy'))
X_val = np.load(os.path.join(save_path, 'val_X.npy'))
y_val = np.load(os.path.join(save_path, 'val_y.npy'))
X_test = np.load(os.path.join(save_path, 'test_X.npy'))
y_test = np.load(os.path.join(save_path, 'test_y.npy'))

num_classes = len(np.unique(y_train))  # 10

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


# Model architecture
def create_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.)(inputs)  # if zero padding exists, though we have fixed length
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)

    # Self-attention layer
    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(128 * 2)(attention)
    attention = layers.Permute([2, 1])(attention)
    sent_representation = layers.Multiply()([x, attention])
    sent_representation = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(sent_representation)

    x = layers.Dense(64, activation='relu')(sent_representation)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model


input_shape = (X_train.shape[1], X_train.shape[2])  # (120, 258)
model = create_model(input_shape, num_classes)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Class weights (if data is imbalanced)
cw = class_weight.compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)),
                                       y=np.argmax(y_train, axis=1))
class_weight_dict = dict(enumerate(cw))

# Callbacks
checkpoint = callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Save final model
model.save('sign_language_model.h5')

# ---------------------------------------------------------
# Plot training/validation accuracy and loss, save as PNG
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

epochs_ran = range(1, len(history.history['accuracy']) + 1)

# Accuracy subplot
axes[0].plot(epochs_ran, history.history['accuracy'], label='Train Accuracy', marker='o')
axes[0].plot(epochs_ran, history.history['val_accuracy'], label='Val Accuracy', marker='o')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss subplot
axes[1].plot(epochs_ran, history.history['loss'], label='Train Loss', marker='o')
axes[1].plot(epochs_ran, history.history['val_loss'], label='Val Loss', marker='o')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f"Training History (Test Accuracy: {test_acc:.4f})")
plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("Saved training_history.png")
plt.show()