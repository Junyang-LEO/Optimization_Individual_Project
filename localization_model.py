import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================================================================
# 1. Custom Dataset Class
# ==============================================================================
class MultiModalSpectrumDataset(Dataset):
    """
    Load and parse multimodal spectrum data from .t files.
    """
    def __init__(self, t_file_path):
        print(f"Initializing dataset, loading file: {t_file_path}")
        if not os.path.isfile(t_file_path):
            raise FileNotFoundError(f"Error: Data file '{t_file_path}' not found.")
        data = torch.load(t_file_path)
        self.labels = data[..., 1:4]
        spt_flat = data[..., 29:]
        self.spectra = spt_flat.reshape(-1, 3, 36, 9)
        g1_pos = data[..., 8:11]
        g1_ori = data[..., 11:15]
        g2_pos = data[..., 15:18]
        g2_ori = data[..., 18:22]
        g3_pos = data[..., 22:25]
        g3_ori = data[..., 25:29]
        self.metadata = torch.cat([g1_pos, g1_ori, g2_pos, g2_ori, g3_pos, g3_ori], dim=2)
        self.num_samples = self.labels.shape[0]
        print(f"Dataset initialization complete, found {self.num_samples} samples.")
        print(f"  - Spatial Spectrum (Spectra) shape: {self.spectra.shape}")
        print(f"  - Metadata shape: {self.metadata.shape}")
        print(f"  - Labels shape: {self.labels.shape}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        spectrum_sample = self.spectra[index]
        metadata_sample = self.metadata[index]
        label_sample = self.labels[index]
        return spectrum_sample, metadata_sample, label_sample

# ==============================================================================
# 2. Hybrid Neural Network Model Definition
# ==============================================================================
class HybridPositioningModel(nn.Module):
    def __init__(self, metadata_dim=21, output_dim=3):
        super(HybridPositioningModel, self).__init__()
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.metadata_mlp = nn.Sequential(nn.Linear(metadata_dim, 64), nn.ReLU(), nn.Linear(64, 32))
        self.fusion_head = nn.Sequential(
            nn.Linear(64 + 32, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, output_dim)
        )

    def forward(self, spectrum_input, metadata_input):
        cnn_features = self.cnn_branch(spectrum_input)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        metadata_features = self.metadata_mlp(metadata_input)
        fused_features = torch.cat((cnn_features, metadata_features), dim=1)
        prediction = self.fusion_head(fused_features)
        return prediction

# ==============================================================================
# 3. Training and Evaluation Functions
# ==============================================================================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training Epoch", leave=False)
    
    for spectra, metadata, labels in progress_bar:
        metadata = metadata.squeeze(1)
        labels = labels.squeeze(1)
        
        spectra = spectra.to(device)
        metadata = metadata.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(spectra, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    return avg_loss

def evaluate_one_epoch(model, data_loader, criterion, device, desc='Evaluating'):
    model.eval()
    total_loss = 0
    total_dist_error = 0
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=desc, leave=False)
        for spectra, metadata, labels in progress_bar:
            spectra = spectra.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)

            metadata = metadata.squeeze(1)
            labels = labels.squeeze(1)

            outputs = model(spectra, metadata)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            dist_error = torch.sqrt(torch.sum((outputs - labels) ** 2, dim=1)).mean()
            total_dist_error += dist_error.item()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}", dist_error=f"{dist_error.item():.4f}")

    avg_loss = total_loss / len(data_loader)
    avg_dist_error = total_dist_error / len(data_loader)
    return avg_loss, avg_dist_error

# ==============================================================================
# 4. Main Program (including final evaluation step)
# ==============================================================================
if __name__ == "__main__":
    # --- Hyperparameters and path settings ---
    TRAIN_VAL_FILE_PATH = "train_0.t"
    TEST_FILE_PATH = "test_0.t"  # This file will be used for final evaluation

    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 25
    VALIDATION_SPLIT_RATIO = 0.2
    
    # --- Device selection ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=========================================")
    print(f"Using device: {DEVICE}")
    print(f"=========================================\n")

    # ==========================================================================
    # Step 1: Compare different optimizers and select the best configuration
    # (on training and validation sets)
    # ==========================================================================
    print("##### Step 1: Optimizer Comparison and Selection #####\n")
    
    # --- Data preparation (executed only once) ---
    print("--- Loading and splitting training and validation sets ---")
    train_val_dataset = MultiModalSpectrumDataset(t_file_path=TRAIN_VAL_FILE_PATH)
    val_size = int(VALIDATION_SPLIT_RATIO * len(train_val_dataset))
    train_size = len(train_val_dataset) - val_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])
    
    print(f"\n  - Training set: {len(train_dataset)} samples")
    print(f"  - Validation set: {len(val_dataset)} samples\n")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    criterion = nn.MSELoss()

    optimization_configs = {
        "SGD_Momentum": {
            "optimizer": lambda params: optim.SGD(params, lr=LEARNING_RATE, momentum=0.9),
            "scheduler": lambda opt: None
        },
        "Adam": {
            "optimizer": lambda params: optim.Adam(params, lr=LEARNING_RATE),
            "scheduler": lambda opt: None
        },
        "AdamW": {
            "optimizer": lambda params: optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.01),
            "scheduler": lambda opt: None
        },
        "Adam_with_ReduceLROnPlateau": {
            "optimizer": lambda params: optim.Adam(params, lr=LEARNING_RATE),
            "scheduler": lambda opt: optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.2, patience=3, verbose=True)
        }
    }

    results = defaultdict(dict)
    best_overall_val_error = float('inf')
    best_optimizer_name = None

    for name, config in optimization_configs.items():
        print("\n" + "="*60)
        print(f"Starting experiment: {name}")
        print("="*60)
        
        model = HybridPositioningModel().to(DEVICE)
        optimizer = config["optimizer"](model.parameters())
        scheduler = config["scheduler"](optimizer)
        
        history = {'train_loss': [], 'val_loss': [], 'val_dist_error': []}
        start_time = time.time()
        
        for epoch in range(NUM_EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
            
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            history['train_loss'].append(train_loss)
            
            val_loss, val_dist_error = evaluate_one_epoch(model, val_loader, criterion, DEVICE, desc='Validating')
            history['val_loss'].append(val_loss)
            history['val_dist_error'].append(val_dist_error)
            
            if val_dist_error < best_overall_val_error:
                best_overall_val_error = val_dist_error
                best_optimizer_name = name

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif scheduler is not None:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS:02d}] Summary - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dist Error: {val_dist_error:.4f} m, LR: {current_lr:.6f}")
        
        end_time = time.time()
        results[name]['history'] = history
        results[name]['time'] = end_time - start_time
        print(f"\nExperiment '{name}' completed, total time: {results[name]['time']:.2f} seconds")

    print(f"\n--- Optimizer comparison completed ---")
    print(f"Best optimizer on validation set: '{best_optimizer_name}' (lowest distance error: {best_overall_val_error:.4f} m)")

    # --- Statistics for key metrics of each optimizer (for richer comparison and visualization) ---
    summary_stats = {}
    for name, result in results.items():
        history = result['history']
        val_dist_errors = np.array(history['val_dist_error'])
        best_idx = int(val_dist_errors.argmin())
        summary_stats[name] = {
            "best_epoch": best_idx + 1,
            "best_val_dist_error": float(val_dist_errors[best_idx]),
            "final_val_loss": float(history['val_loss'][-1]),
            "time": float(result['time'])
        }
    
    print("\n--- Summary of Key Metrics for Each Optimizer (Validation Set) ---")
    for name, stats in summary_stats.items():
        print(f"[{name}] "
              f"Best Epoch: {stats['best_epoch']:02d}, "
              f"Best Avg Distance Error: {stats['best_val_dist_error']:.4f} m, "
              f"Final Val Loss: {stats['final_val_loss']:.4f}, "
              f"Total Training Time: {stats['time']:.2f} s")

    # --- Result visualization ---
    print("\n--- Generating comparison plots... ---")
    # Set a beautiful plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a figure with 3 subplots sharing the X-axis
    # figsize controls the size of the entire figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18), sharex=True)
    
    # Set a main title for the entire figure
    fig.suptitle('Optimizer Performance Comparison on Validation Set', fontsize=20)
    
    # Get a color map to ensure each optimizer's curve has a different color
    colors = plt.cm.get_cmap('tab10', len(results))
    
    # Iterate through the results dictionary, each key-value pair represents an optimizer's experimental results
    for i, (name, result) in enumerate(results.items()):
        # Extract history from results
        history = result['history']
        # Create X-axis range, i.e., the number of epochs
        epochs_range = range(1, NUM_EPOCHS + 1)
        
        # Plot training loss curve on the first subplot (ax1)
        ax1.plot(epochs_range, history['train_loss'], label=f'{name}', color=colors(i), marker='o', markersize=4, alpha=0.8)
        
        # Plot validation loss curve on the second subplot (ax2)
        ax2.plot(epochs_range, history['val_loss'], label=f'{name}', color=colors(i), marker='x', markersize=5, alpha=0.8)
        
        # Plot validation distance error curve on the third subplot (ax3)
        ax3.plot(epochs_range, history['val_dist_error'], label=f'{name}', color=colors(i), marker='s', markersize=4, alpha=0.8)

    # Set title, Y-axis label, legend, and grid for the first subplot
    ax1.set_title("Training Loss (MSE)", fontsize=16)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--')

    # Set title, Y-axis label, legend, and grid for the second subplot
    ax2.set_title("Validation Loss (MSE)", fontsize=16)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--')

    # Set title, X/Y-axis labels, legend, and grid for the third subplot
    ax3.set_title("Validation Average Distance Error (meters)", fontsize=16)
    ax3.set_xlabel("Epochs", fontsize=14)
    ax3.set_ylabel("Distance Error (m)", fontsize=12)
    ax3.legend()
    ax3.grid(True, which='both', linestyle='--')
    
    # Set X-axis ticks to make them more readable, avoiding overly dense ticks
    plt.xticks(np.arange(1, NUM_EPOCHS + 1, step=max(1, NUM_EPOCHS // 10)))
    
    # Automatically adjust subplot layout to prevent title and label overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save the main figure containing loss and error curves
    fig.savefig('opt_withtest_results.png', dpi=300, bbox_inches='tight')

    # ----------------------------------------------------------------------
    # Additional Visualization 1: Bar chart comparison of "Best Validation Distance Error" 
    # and "Total Training Time" for each optimizer
    # ----------------------------------------------------------------------
    fig2, (ax4, ax5) = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle('Best Validation Distance Error & Training Time by Optimizer', fontsize=18)

    opt_names = list(summary_stats.keys())
    best_errors = [summary_stats[n]['best_val_dist_error'] for n in opt_names]
    times = [summary_stats[n]['time'] for n in opt_names]

    x = np.arange(len(opt_names))

    # Left: Best validation distance error
    ax4.bar(x, best_errors, color='tab:blue', alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(opt_names, rotation=20)
    ax4.set_ylabel('Best Val Distance Error (m)')
    ax4.set_title('Best Validation Distance Error')

    # Right: Total training time
    ax5.bar(x, times, color='tab:orange', alpha=0.8)
    ax5.set_xticks(x)
    ax5.set_xticklabels(opt_names, rotation=20)
    ax5.set_ylabel('Total Training Time (s)')
    ax5.set_title('Training Time Comparison')

    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the second figure (bar chart summary)
    fig2.savefig('optimizer_summary.png', dpi=300, bbox_inches='tight')

    # Display all generated charts
    plt.show()


    # ==========================================================================
    # Step 2: Use the selected best configuration for final evaluation on 
    # independent test set
    # ==========================================================================
    print("\n\n" + "#"*60)
    print("##### Step 2: Final Model Evaluation #####")
    print("#"*60 + "\n")

    if best_optimizer_name is None:
        print("No best optimizer found, skipping final evaluation.")
    else:
        print(f"--- Retraining final model using selected best configuration '{best_optimizer_name}' ---")
        
        # 1. Reinitialize model and optimizer
        final_model = HybridPositioningModel().to(DEVICE)
        final_config = optimization_configs[best_optimizer_name]
        final_optimizer = final_config["optimizer"](final_model.parameters())
        final_scheduler = final_config["scheduler"](final_optimizer)
        
        # 2. Retrain model and find best checkpoint on validation set
        best_val_loss = float('inf')
        model_save_path = "final_best_model.pth"
        
        print(f"Will train for {NUM_EPOCHS} epochs using '{best_optimizer_name}' and save the model with lowest validation loss.")
        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(final_model, train_loader, criterion, final_optimizer, DEVICE)
            val_loss, val_dist_error = evaluate_one_epoch(final_model, val_loader, criterion, DEVICE, desc='Validating for final model')
            
            print(f"Final Training - Epoch [{epoch+1:02d}/{NUM_EPOCHS:02d}] | Val Loss: {val_loss:.4f} | Val Dist Error: {val_dist_error:.4f} m")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(final_model.state_dict(), model_save_path)
                print(f"  -> Validation loss decreased, best model saved to '{model_save_path}'")
            
            if isinstance(final_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                final_scheduler.step(val_loss)
            elif final_scheduler is not None:
                final_scheduler.step()
        
        print("\n--- Final model training completed ---")

        # 3. Load independent test set
        print("\n--- Loading independent test set for final evaluation ---")
        test_dataset = MultiModalSpectrumDataset(t_file_path=TEST_FILE_PATH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        print(f"  - Test set: {len(test_dataset)} samples (from '{os.path.basename(TEST_FILE_PATH)}')\n")

        # 4. Load best model and evaluate on test set
        print(f"--- Loading '{model_save_path}' and running on test set ---")
        model_for_testing = HybridPositioningModel().to(DEVICE)
        model_for_testing.load_state_dict(torch.load(model_save_path))
        
        test_loss, test_dist_error = evaluate_one_epoch(model_for_testing, test_loader, criterion, DEVICE, desc='FINAL TESTING')
        
        print("\n" + "-"*50)
        print("Final model performance on independent test set:")
        print(f"  - Test Loss (MSE): {test_loss:.4f}")
        print(f"  - Test Average Distance Error: {test_dist_error:.4f} m")
        print("-"*50)

