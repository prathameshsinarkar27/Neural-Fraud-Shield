"""
models/dnn_model.py
==============================================================================
Deep Neural Network definition and training for fraud detection.
  Input(30) -> Dense(128) -> BN -> ReLU -> Dropout(0.3)
            -> Dense(64)  -> BN -> ReLU -> Dropout(0.4)
            -> Dense(32)  -> BN -> ReLU -> Dropout(0.3)
            -> Dense(16)  -> ReLU
            -> Dense(1, sigmoid)
  Optimizer: Adam (lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
  Loss: Binary Cross-Entropy
  Training: Backpropagation via Keras .fit()
==============================================================================
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, callbacks

BATCH_SIZE = 256
EPOCHS = 30


def build_dnn_model():
    """Build and compile the Fraud_Detection_DNN model. Returns the compiled model."""
    print("\n>>> ARCHITECTURE: Deep Neural Network for Binary Classification")
    print(">>> Activation: ReLU (hidden layers), Sigmoid (output layer)")
    print(">>> Regularization: Dropout (0.3-0.4) + Batch Normalization")

    model = keras.Sequential([
        # Input Layer
        layers.Input(shape=(30,), name='input_layer'),

        # Hidden Layer 1: 128 neurons
        layers.Dense(128, name='dense_1'),
        layers.BatchNormalization(name='batch_norm_1'),
        layers.Activation('relu', name='relu_1'),
        layers.Dropout(0.3, name='dropout_1'),

        # Hidden Layer 2: 64 neurons
        layers.Dense(64, name='dense_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Activation('relu', name='relu_2'),
        layers.Dropout(0.4, name='dropout_2'),

        # Hidden Layer 3: 32 neurons
        layers.Dense(32, name='dense_3'),
        layers.BatchNormalization(name='batch_norm_3'),
        layers.Activation('relu', name='relu_3'),
        layers.Dropout(0.3, name='dropout_3'),

        # Hidden Layer 4: 16 neurons
        layers.Dense(16, name='dense_4'),
        layers.Activation('relu', name='relu_4'),

        # Output Layer: Sigmoid for binary classification
        layers.Dense(1, activation='sigmoid', name='output_layer')
    ], name='Fraud_Detection_DNN')

    model.summary()

    # ----- OPTIMIZER: ADAM -----
    print("\n>>> OPTIMIZER: Adam (Adaptive Moment Estimation)")
    print(">>> Adam uses exponential moving averages of gradients (m_t) and")
    print(">>> squared gradients (v_t) to adapt the learning rate per parameter.")
    print(">>> Learning Rate: 0.001 (default)")
    print(">>> Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-7")

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    # ----- LOSS FUNCTION: Binary Cross-Entropy -----
    print("\n>>> LOSS FUNCTION: Binary Cross-Entropy")
    print(">>> Measures the difference between predicted probability and actual label")

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    return model


def get_architecture_info(model):
    """Per-layer architecture metadata, saved into training_history.json for the frontend."""
    architecture_info = []
    for layer in model.layers:
        layer_info = {
            'name': layer.name,
            'type': layer.__class__.__name__,
            'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A',
            'params': int(layer.count_params())
        }
        architecture_info.append(layer_info)
    return architecture_info


def train_dnn(model, X_train_smote, y_train_smote, X_test_scaled, y_test):
    """Train the DNN via backpropagation (Keras .fit()). Returns (history, best_epoch).

    Keras .fit() performs BACKPROPAGATION automatically:
      1. Forward Pass: Input flows through layers to produce predictions
      2. Loss Computation: Binary cross-entropy measures prediction error
      3. Backward Pass: Gradients of loss w.r.t. each weight are computed
         via the chain rule (this IS backpropagation)
      4. Weight Update: Adam optimizer updates weights using computed gradients
    """
    print("\n>>> TRAINING MECHANISM: Backpropagation (via Keras .fit())")
    print(">>> The .fit() method implements the backpropagation algorithm:")
    print(">>>   1. Forward Pass: Input → Dense → BatchNorm → ReLU → Dropout → Output")
    print(">>>   2. Loss Calculation: Binary Cross-Entropy between predictions and labels")
    print(">>>   3. Backward Pass (BACKPROP): Chain rule computes ∂Loss/∂weights for each layer")
    print(">>>   4. Weight Update: Adam optimizer applies gradient-based updates")
    print(">>> This process repeats for each batch in each epoch.\n")

    early_stop = callbacks.EarlyStopping(
        monitor='val_auc', patience=5, mode='max', restore_best_weights=True,
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    total_params = int(model.count_params())
    print(f">>> Training DNN with Batch Size: {BATCH_SIZE}, Max Epochs: {EPOCHS}")
    print(f">>> Early Stopping: patience=5, monitoring val_auc")
    print(f">>> Total trainable parameters: {total_params:,}")
    print(">>> Starting backpropagation training loop...\n")

    history = model.fit(
        X_train_smote, y_train_smote,
        validation_data=(X_test_scaled, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    best_epoch = int(np.argmax(history.history['val_auc'])) + 1
    print(f"\n>>> BACKPROPAGATION COMPLETE. Best epoch: {best_epoch}")
    print(f">>> Adam Optimizer successfully minimized the loss via gradient descent.")

    return history, best_epoch


def predict_dnn(model, X):
    """Flattened DNN probability predictions for a batch of scaled features."""
    return model.predict(X, verbose=0).flatten()
