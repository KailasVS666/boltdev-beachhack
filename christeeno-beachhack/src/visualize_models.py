import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
from src.config import ARTIFACT_DIR
# Torch and Models will be imported lazily to avoid DLL issues on Windows

def plot_sfi_results(df_original, df_fault, smooth_signal, output_dir="."):
    """
    Plots the SFI injection results: Original vs Fault vs Smoothed.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot raw faulty signal
    plt.plot(df_fault['FLAP'].values[:200], label='Fault Injected (Raw)', color='red', alpha=0.5, linestyle='--')
    
    # Plot smoothed signal
    plt.plot(smooth_signal.values[:200], label='JRR Smoothed (Filtered)', color='green', linewidth=2)
    
    # Plot original baseline (if available, or infer from fault-noise)
    # Ideally we'd pass df_clean, but here we show the effect
    if df_original is not None:
         plt.plot(df_original['FLAP'].values[:200], label='Baseline (Normal)', color='blue', alpha=0.3)
    
    plt.title("SFI Validation: Flap Actuator Oscillation & JRR Filtering")
    plt.ylabel("Flap Angle (deg)")
    plt.xlabel("Time (frames)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, "sfi_validation_plot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"   -> [VIS] Saved SFI Plot: {save_path}")

def plot_model_anomaly(scores, threshold=0.0, output_dir="."):
    """
    Plots Isolation Forest Anomaly Scores.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(scores, label='Anomaly Score', color='purple')
    plt.axhline(y=threshold, color='red', linestyle='--', label='Decision Boundary')
    plt.title("Isolation Forest Anomaly Scores")
    plt.ylabel("Score (Negative = Anomaly)")
    plt.xlabel("Time Step")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(output_dir, "model_anomaly_score.png")
    plt.savefig(save_path)
    plt.close()
    print(f"   -> [VIS] Saved Anomaly Score Plot: {save_path}")

def plot_lstm_rul(predictions, quantiles=[0.05, 0.5, 0.95], output_dir="."):
    """
    Plots LSTM RUL predictions with uncertainty bounds.
    predictions: [seq_len, 3] (low, median, high)
    """
    plt.figure(figsize=(12, 5))
    
    # Assuming predictions are numpy array: [time, quantiles]
    t = np.arange(len(predictions))
    
    med = predictions[:, 1]
    low = predictions[:, 0]
    high = predictions[:, 2]
    
    plt.plot(t, med, label='Predicted RUL (Median)', color='blue')
    plt.fill_between(t, low, high, color='blue', alpha=0.2, label='Uncertainty (5-95%)')
    
    plt.title("LSTM Remaining Useful Life (RUL) Prediction")
    plt.ylabel("RUL (Cycles/Flights)")
    plt.xlabel("Time Step")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(output_dir, "lstm_rul_prediction.png")
    plt.savefig(save_path)
    plt.close()
    print(f"   -> [VIS] Saved LSTM RUL Plot: {save_path}")

def plot_ae_reconstruction(dates, reconstruction_errors, threshold=None, output_dir="."):
    """
    Plots Autoencoder Reconstruction Error over time.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(reconstruction_errors, label='Reconstruction MSE', color='orange')
    if threshold:
        plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
        
    plt.title("Autoencoder Reconstruction Error (Safety Interlock)")
    plt.ylabel("MSE Loss")
    plt.xlabel("Time Sequence")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(output_dir, "ae_reconstruction_error.png")
    plt.savefig(save_path)
    plt.close()
    print(f"   -> [VIS] Saved AE Reconstruction Plot: {save_path}")

def generate_all_visuals(df_original, df_fault, smoothed_signal, hybrid_model, scaler, output_dir="."):
    """
    Master function to run all visualizations.
    """
    print("\n[VIS] Generating Model Visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. SFI Plot
    plot_sfi_results(df_original, df_fault, smoothed_signal, output_dir)
    
    # 2. IForest Scores (on Faulty Data)
    if hybrid_model.iforest:
        print("   -> Running Isolation Forest on sequence...")
        # Transform data
        X = df_fault[scaler.feature_names_in_].values
        X_scaled = scaler.transform(X)
        scores = hybrid_model.iforest.decision_function(X_scaled)
        plot_model_anomaly(scores, output_dir=output_dir)
        
    # 3. LSTM & AE (if loaded)
    # Load Models if not passed in fully
    try:
        import torch
        from src.models import BiDirQuantileLSTM, SimpleAutoencoder
        
        # Load LSTM
        lstm_path = os.path.join(ARTIFACT_DIR, "aeroguard_v1_lstm.pth")
        if os.path.exists(lstm_path):
             print(f"   -> Loading LSTM from {lstm_path}...")
             lstm = BiDirQuantileLSTM(len(scaler.feature_names_in_))
             lstm.load_state_dict(torch.load(lstm_path))
             lstm.eval()
             
             # Create sequences
             seq_len = 50 # Assumption
             X = df_fault[scaler.feature_names_in_].values
             X_scaled = scaler.transform(X)
             X_tensor = torch.FloatTensor(X_scaled)
             
             probs = []
             # Sliding window inference (simplified: every 10th frame for speed)
             for i in range(seq_len, len(X_tensor), 10):
                 seq = X_tensor[i-seq_len:i].unsqueeze(0) # [1, seq, feat]
                 with torch.no_grad():
                     output = lstm(seq)
                 probs.append(output.numpy()[0])
             
             plot_lstm_rul(np.array(probs), output_dir=output_dir)
        
        # Load AE
        ae_path = os.path.join(ARTIFACT_DIR, "aeroguard_v1_ae.pth")
        if os.path.exists(ae_path):
             print(f"   -> Loading Autoencoder from {ae_path}...")
             ae = SimpleAutoencoder(len(scaler.feature_names_in_))
             ae.load_state_dict(torch.load(ae_path))
             ae.eval()
             
             X = df_fault[scaler.feature_names_in_].values
             X_scaled = scaler.transform(X)
             X_tensor = torch.FloatTensor(X_scaled)
             
             errors = []
             for i in range(len(X_tensor)):
                 frame = X_tensor[i].unsqueeze(0)
                 with torch.no_grad():
                     recon = ae(frame)
                 loss = torch.mean((frame - recon)**2).item()
                 errors.append(loss)
                 
             plot_ae_reconstruction(None, errors, threshold=0.1, output_dir=output_dir)
             
    except Exception as e:
        print(f"   -> [VIS WARNING] Could not run Deep Learning visualizers: {e}")
