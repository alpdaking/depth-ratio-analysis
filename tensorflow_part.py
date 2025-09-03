import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dropout, Activation, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import pickle

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    epsilon = 1e-4
    y_true_safe = tf.clip_by_value(y_true, epsilon, float('inf'))
    
    percentage_errors = tf.abs((y_true_safe - y_pred) / y_true_safe) * 100
    
    max_percentage = 1000.0
    percentage_errors_clipped = tf.clip_by_value(percentage_errors, 0.0, max_percentage)
    
    return tf.reduce_mean(percentage_errors_clipped)

def preprocess(data_raw, seq_len, train_split, val_split, scaler=None, fit=True):
    """
    Creates sequences, splits into train/val/test, and scales.
    Scaling is FIT ONLY on X_train, then applied to X_val, X_test, and y_*.

    Args:
        data_raw (np.array): Raw feature array [num_rows, num_features], unscaled.
        seq_len (int): Total sequence length window used to build (X,y).
        train_split (float): Proportion of sequences used for training.
        val_split (float): Proportion of sequences used for validation.
        scaler (MinMaxScaler | None): Provide to reuse; if None and fit=True, a new scaler is created.
        fit (bool): If True, fit scaler on X_train only. If False, just transform with provided scaler.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
    """
    def to_sequences(data, seq_len):
        d = []
        for i in range(len(data) - seq_len):
            d.append(data[i: i + seq_len])
        return np.array(d)

    # Build sequences on raw (unscaled) data
    data = to_sequences(data_raw, seq_len)
    num_train = int(train_split * data.shape[0])
    num_val = int(val_split * data.shape[0])

    # X: first (seq_len - 10) steps, y: the step at -10
    X_train = data[:num_train, :-10, :]
    y_train = data[:num_train, -10, :]

    X_val   = data[num_train:num_train + num_val, :-10, :]
    y_val   = data[num_train:num_train + num_val, -10, :]

    X_test  = data[num_train + num_val:, :-10, :]
    y_test  = data[num_train + num_val:, -10, :]  # fixed vs. original

    # Fit-on-train-only scaling
    n_features = X_train.shape[-1]
    if scaler is None:
        scaler = MinMaxScaler()

    if fit:
        scaler.fit(X_train.reshape(-1, n_features))

    # Transform X sets
    X_train = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val   = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test  = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    # Transform y sets (shape [n_samples, n_features])
    y_train = scaler.transform(y_train)
    y_val   = scaler.transform(y_val)
    y_test  = scaler.transform(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

def numpy_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    print(f"MAPE Debug - y_true range: {np.min(y_true):.6f} to {np.max(y_true):.6f}")
    print(f"MAPE Debug - y_pred range: {np.min(y_pred):.6f} to {np.max(y_pred):.6f}")
    print(f"MAPE Debug - y_true mean: {np.mean(y_true):.6f}")
    print(f"MAPE Debug - y_pred mean: {np.mean(y_pred):.6f}")
    
    epsilon = 1e-6
    y_true_safe = np.clip(y_true, epsilon, None)
    
    percentage_errors = np.abs((y_true_safe - y_pred) / y_true_safe) * 100
    
    max_percentage = 1000.0
    percentage_errors_clipped = np.clip(percentage_errors, 0.0, max_percentage)
    
    print(f"MAPE Debug - Percentage errors range: {np.min(percentage_errors_clipped):.2f}% to {np.max(percentage_errors_clipped):.2f}%")
    print(f"MAPE Debug - Percentage errors mean: {np.mean(percentage_errors_clipped):.2f}%")
    print(f"MAPE Debug - Number of clipped values: {np.sum(percentage_errors > max_percentage)}")
    
    return np.mean(percentage_errors_clipped)

def midprice_mse(y_true, y_pred):
    y_true_mid = y_true[..., -1]
    y_pred_mid = y_pred[..., -1]
    return tf.reduce_mean(tf.square(y_true_mid - y_pred_mid))

def main():
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED) 

    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    mps_device_name = "/device:GPU:0"
    if tf.config.list_physical_devices('GPU'):
        print("\n detected GPU.")
        try:
            tf.config.set_logical_device_configuration(
                tf.config.list_physical_devices('GPU')[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
            )
            print("using GPU .")
        except RuntimeError as e:
            print(e)
        
        device_name = mps_device_name
    else:
        print("\nNo GPU .")
        device_name = "/device:CPU:0"
        
    print(f"Using device: {device_name}")  

    experiment_name = 'short_improvements_pat20'
    print("Starting experiment: " + experiment_name)

    exp_root = os.path.join("experiments", experiment_name)
    logs_dir = os.path.join("logs", experiment_name)
    graphs_dir = os.path.join(exp_root, "graphs")
    ckpt_path = os.path.join(exp_root, "model.h5")
    results_pkl = os.path.join(exp_root, "tensorflow_results.pkl")

    os.makedirs(exp_root, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
 
    data_dir = "data"
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {file}")
    
    dfs = []
    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        print(f"Reading {csv_file}...")
        temp_df = pd.read_csv(csv_path, sep=';', parse_dates=['DateTime'])
        dfs.append(temp_df)
        print(f"  - Shape: {temp_df.shape}")
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined DataFrame shape: {df.shape}")
    
    df = df.sort_values('DateTime').reset_index(drop=True)
    print(f"DataFrame sorted by DateTime, final shape: {df.shape}")

    df['midPrice'] = (df['Level 1 Bid Price'] + df['Level 1 Ask Price']) / 2
    
    feature_columns = [
        'Depth Ratio',
        'Last Price', 
        'Total Bid Volume',
        ' Total Ask Volume',
        'Level 1 Bid Price',
        'Level 1 Bid Volume',
        'Level 1 Ask Price', 
        'Level 1 Ask Volume',
        'Level 2 Bid Price',
        'Level 2 Bid Volume',
        'Level 2 Ask Price', 
        'Level 2 Ask Volume',
        'Level 3 Bid Price',
        'Level 3 Bid Volume',
        'Level 3 Ask Price', 
        'Level 3 Ask Volume',
        'Level 4 Bid Price',
        'Level 4 Bid Volume',
        'Level 4 Ask Price', 
        'Level 4 Ask Volume',
        'Level 5 Bid Price',
        'Level 5 Bid Volume',
        'Level 5 Ask Price', 
        'Level 5 Ask Volume',
        'midPrice'
    ]
    
    target_column = 'midPrice'
    
    print(f"\nSelected Features: {feature_columns}")
    print(f"Target Column: {target_column}")
    
    feature_data = df[feature_columns].values
    
    if np.isnan(feature_data).any():
        print("Warning: NaN values found in feature data. Forward-filling them.")
        feature_data = pd.DataFrame(feature_data, columns=feature_columns).fillna(method='ffill').values
    
    SEQ_LEN = 300
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess(feature_data, SEQ_LEN, train_split=0.5, val_split=0.1)

    print("1")
        
    DROPOUT = 0.2
    WINDOW_SIZE = SEQ_LEN - 10
    N_FEATURES = X_train.shape[-1]

    model = keras.Sequential()
    model.add(LSTM(WINDOW_SIZE, return_sequences=True, input_shape=(WINDOW_SIZE, N_FEATURES)))
    model.add(Dropout(DROPOUT))
    model.add(LSTM(WINDOW_SIZE, return_sequences=True))
    model.add(Dropout(rate=DROPOUT))
    model.add(LSTM(WINDOW_SIZE, return_sequences=False)) 
    model.add(Dropout(rate=DROPOUT))
    model.add(Dense(units=N_FEATURES))
    model.add(Activation('linear'))

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6
    )
    
    model_checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=20,
        verbose=1,
        restore_best_weights=True,
    )

    tb_callback = TensorBoard(
        log_dir=logs_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        update_freq='epoch',
        profile_batch=0         
    )

    adam = Adam(learning_rate=1e-4)
    model.compile(loss=midprice_mse, optimizer=adam, metrics=[mean_absolute_percentage_error])
    
    BATCH_SIZE = 300

    print("\nStarting model training...")
    try:
        history = model.fit(
            X_train,
            y_train,
            epochs=1,
            batch_size=BATCH_SIZE,
            shuffle=False, 
            validation_data=(X_val, y_val),
            callbacks=[lr_scheduler, model_checkpoint_callback, tb_callback, early_stopping_callback]
        )
        print("Model training finished.")
    except Exception as e:
        print(f"Error during model training: {e}")
        print("Attempting to continue with existing model...")
        # Create a dummy history object
        history = type('obj', (object,), {
            'history': {
                'loss': [0.1],
                'val_loss': [0.1]
            }
        })()
    
    if os.path.exists(ckpt_path):
        print(f"\nLoading best model from {ckpt_path} for evaluation.")
        best_model = keras.models.load_model(ckpt_path, 
                                            custom_objects={'mean_absolute_percentage_error': mean_absolute_percentage_error,
                                            'midprice_mse': midprice_mse               
                                            })
    else:
        print(f"\nError: Best model not found at {ckpt_path}. Using the last trained model.")
        best_model = model 

    print("\nEvaluating model on test data...")
    test_loss, test_mape = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (best model): {test_loss:.6f}")
    print(f"Test MAPE (best model): {test_mape:.2f}%")
    
    try:
        plt.figure(figsize=(12, 4))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(graphs_dir, 'training_loss.png'))
        plt.show()
    except Exception as e:
        print(f"Warning: Could not create training loss plot: {e}")
        print("Continuing with analysis...")
    
    print("\nMaking predictions on test data...")
    y_hat = best_model.predict(X_test)

    y_test_inverse = scaler.inverse_transform(y_test)
    y_hat_inverse = scaler.inverse_transform(y_hat)
    
    last_col_idx = feature_columns.index('midPrice') 
    y_test_last = y_test_inverse[:, last_col_idx]
    y_hat_last = y_hat_inverse[:, last_col_idx]

    try:
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_last, label="Actual Mid Price", color='green', alpha=0.7)
        plt.plot(y_hat_last, label="Predicted Mid Price", color='red', alpha=0.7)

        plt.title('Mid Price Prediction - Multivariate LSTM')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(graphs_dir, 'prediction_comparison.png'))
        plt.show()
    except Exception as e:
        print(f"Warning: Could not create prediction comparison plot: {e}")
        print("Continuing with analysis...")
    
    mse = np.mean((y_test_last - y_hat_last) ** 2)
    mae = np.mean(np.abs(y_test_last - y_hat_last))
    rmse = np.sqrt(mse)
    mape = numpy_mape(y_test_last, y_hat_last)

    print(f"\nPrediction Metrics (on inverse transformed 'midPrice'):")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAPE: {mape:.2f}%")
    
    print(f"\nFeature columns used for training: {feature_columns}")
    print(f"Number of features: {N_FEATURES}")

    # Calculate start index for trading decisions
    # With new split: train 50%, validation 10%, test 40%
    # start_index = SEQ_LEN + train_size + val_size
    train_size = int(0.5 * (len(df) - SEQ_LEN))
    val_size = int(0.1 * (len(df) - SEQ_LEN))
    start_index = SEQ_LEN + train_size + val_size
    
    print(f"Data split - Train: 50%, Validation: 10%, Test: 40%")
    print(f"Start index for trading decisions: {start_index}")

    # Save results for the next step
    results = {
        'df': df,
        'y_hat_last': y_hat_last,
        'y_test_last': y_test_last,
        'SEQ_LEN': SEQ_LEN,
        'feature_columns': feature_columns,
        'scaler': scaler,
        'start_index': start_index,
        'metrics': {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    }
    
    # Save results to file
    with open(results_pkl, 'wb') as f:
        pickle.dump(results, f)
     
    print("\nTensorFlow results saved to 'tensorflow_results.pkl'")
    print("TensorFlow part completed successfully!")

if __name__ == "__main__":
    main()
