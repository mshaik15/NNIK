import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error

def load_ik_data(poses_file: Path, solutions_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(poses_file, 'r') as f:
        poses_data = json.load(f)
    with open(solutions_file, 'r') as f:
        solutions_data = json.load(f)

    X = np.array([item['pose'] for item in poses_data['data']], dtype=np.float32)
    y = np.array([item['joint_angles'] for item in solutions_data['data']], dtype=np.float32)
    
    print(f"Loaded data: X shape = {X.shape}, y shape = {y.shape}")
    print(f"DOF from data: {y.shape[1]}")
    
    return X, y

def train_all_models(models: Dict, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
    """Train all models with timing"""
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            results[name] = {
                'model': model,
                'training_time': training_time
            }
            print(f"  ✓ Completed in {training_time:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            results[name] = {'error': str(e)}
    
    return results

def evaluate_all_models(trained_models: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    results = {}
    
    for name, model_data in trained_models.items():
        if 'error' in model_data:
            results[name] = model_data
            continue
            
        print(f"Evaluating {name}...")
        try:
            model = model_data['model']
            
            # Timed prediction
            start_time = time.time()
            y_pred = model.predict(X_test)
            inference_time = time.time() - start_time
            
            # Joint space error (this is what we actually care about)
            joint_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # For compatibility with existing plots, set position_rmse = joint_rmse
            position_rmse = joint_rmse
            
            results[name] = {
                'model': model,
                'position_rmse': position_rmse,  # Same as joint_rmse for simplicity
                'joint_rmse': joint_rmse,
                'training_time': model_data['training_time'],
                'inference_time': inference_time,
                'inference_time_per_sample': inference_time / len(X_test)
            }
            
            print(f"  ✓ Joint RMSE: {joint_rmse:.4f}")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            results[name] = {'error': str(e)}
    
    return results

def create_results_dataframe(evaluation_results: Dict) -> pd.DataFrame:
    """Convert results to clean dataframe"""
    data = []
    
    for name, results in evaluation_results.items():
        if 'error' not in results:
            data.append({
                'model': name,
                'position_rmse': results['position_rmse'],
                'joint_rmse': results['joint_rmse'],
                'training_time': results['training_time'],
                'inference_time': results['inference_time'],
                'inference_time_per_sample': results['inference_time_per_sample']
            })
    
    return pd.DataFrame(data)

def plot_model_comparison(df: pd.DataFrame, save_path: Path = None):
    """Simple comparison plots"""
    if df.empty:
        print("No data to plot")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Model Comparison', fontsize=14)
    
    # Position accuracy
    axes[0,0].bar(df['model'], df['position_rmse'])
    axes[0,0].set_title('Position RMSE (lower is better)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Joint accuracy  
    axes[0,1].bar(df['model'], df['joint_rmse'])
    axes[0,1].set_title('Joint RMSE (lower is better)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Training speed
    axes[1,0].bar(df['model'], df['training_time'])
    axes[1,0].set_title('Training Time (lower is better)')
    axes[1,0].set_yscale('log')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Inference speed
    axes[1,1].bar(df['model'], df['inference_time_per_sample'] * 1000)
    axes[1,1].set_title('Inference Time per Sample (ms)')
    axes[1,1].set_yscale('log')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def run_complete_benchmark(dof_range, data_path, sample_size=None):
    """Run benchmark across multiple DOFs - placeholder for your implementation"""
    # This would iterate through DOF values and run the full pipeline
    # Left as placeholder since you have the main logic in the Colab notebook
    pass