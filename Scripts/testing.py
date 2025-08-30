import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import torch
from sklearn.metrics import mean_squared_error
from typing import Dict
# Import training utilities
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

try:
    from Scripts.training import load_ik_data, train_all_models
    from Scripts.Models.Machine_Learning import ANNModel, KNNModel, ELMModel, RandomForestModel, SVMModel, GPRModel, MDNModel, CVAEModel
except ImportError:
    try:
        from training import load_ik_data, train_all_models
        from Models.Machine_Learning import ANNModel, KNNModel, ELMModel, RandomForestModel, SVMModel, GPRModel, MDNModel, CVAEModel
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the project root directory")

def evaluate_all_models(trained_models: Dict, X_test: np.ndarray, y_test: np.ndarray, force_cpu: bool = True) -> Dict:
    # Evaluate all models (CPU only for fair comparison)
    results = {}
    
    if force_cpu:
        print("Evaluation using CPU only for fair comparison")
    
    for name, model_data in trained_models.items():
        if 'error' in model_data:
            results[name] = model_data
            continue
            
        print(f"Evaluating {name}...")
        try:
            model = model_data['model']
            
            # Force CPU for inference if requested
            if force_cpu and name in ['ANN', 'MDN', 'CVAE']:
                if hasattr(model, 'device'):
                    original_device = model.device
                    model.device = torch.device('cpu')
                elif hasattr(model, 'to'):
                    model.cpu()
            
            # Timed prediction
            start_time = time.time()
            y_pred = model.predict(X_test)
            inference_time = time.time() - start_time
            
            # Restore original device if changed
            if force_cpu and name in ['ANN', 'MDN', 'CVAE']:
                if hasattr(model, 'device') and 'original_device' in locals():
                    model.device = original_device
            
            # Joint space error (this is what we actually care about)
            joint_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # For compatibility with existing plots, set position_rmse = joint_rmse
            position_rmse = joint_rmse
            
            results[name] = {
                'model': model,
                'position_rmse': position_rmse,
                'joint_rmse': joint_rmse,
                'training_time': model_data['training_time'],
                'inference_time': inference_time,
                'inference_time_per_sample': inference_time / len(X_test)
            }
            
            print(f"  ✓ Joint RMSE: {joint_rmse:.4f}, Inference: {inference_time:.3f}s")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            results[name] = {'error': str(e)}
    
    return results

def create_results_dataframe(evaluation_results: Dict) -> pd.DataFrame:
    # Convert evaluation results to DataFrame
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
    # Plot model comparison charts
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

def plot_scalability_analysis(df, save_path=None):
    # Plot scalability analysis across DOFs
    if df is None or df.empty:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Model Scalability Analysis', fontsize=14)
    
    # 1. Training Time vs DOF
    ax1 = axes[0, 0]
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        ax1.plot(model_data['dof'], model_data['training_time'], 'o-', label=model, linewidth=2)
    ax1.set_xlabel('DOF')
    ax1.set_ylabel('Training Time (s)')
    ax1.set_title('Training Time vs DOF')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Position RMSE vs DOF
    ax2 = axes[0, 1]
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        ax2.plot(model_data['dof'], model_data['position_rmse'], 'o-', label=model, linewidth=2)
    ax2.set_xlabel('DOF')
    ax2.set_ylabel('Position RMSE')
    ax2.set_title('Position Accuracy vs DOF')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Joint RMSE vs DOF
    ax3 = axes[1, 0]
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        ax3.plot(model_data['dof'], model_data['joint_rmse'], 'o-', label=model, linewidth=2)
    ax3.set_xlabel('DOF')
    ax3.set_ylabel('Joint RMSE')
    ax3.set_title('Joint Accuracy vs DOF')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Inference Time vs DOF
    ax4 = axes[1, 1]
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        ax4.plot(model_data['dof'], model_data['inference_time_per_sample']*1000, 'o-', label=model, linewidth=2)
    ax4.set_xlabel('DOF')
    ax4.set_ylabel('Inference Time (ms)')
    ax4.set_title('Inference Speed vs DOF')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_model_ranking(df, save_path=None):
    # Plot model ranking based on performance
    if df is None or df.empty:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Average performance across all DOFs
    avg_metrics = df.groupby('model').agg({
        'position_rmse': 'mean',
        'joint_rmse': 'mean',
        'training_time': 'mean',
        'inference_time_per_sample': 'mean'
    }).round(4)
    
    # 1. Accuracy ranking
    accuracy_score = avg_metrics['position_rmse'].rank() + avg_metrics['joint_rmse'].rank()
    accuracy_ranking = accuracy_score.sort_values()
    
    axes[0].barh(range(len(accuracy_ranking)), accuracy_ranking.values)
    axes[0].set_yticks(range(len(accuracy_ranking)))
    axes[0].set_yticklabels(accuracy_ranking.index)
    axes[0].set_xlabel('Accuracy Rank (lower is better)')
    axes[0].set_title('Model Accuracy Ranking')
    
    # 2. Speed ranking
    speed_score = avg_metrics['training_time'].rank() + avg_metrics['inference_time_per_sample'].rank()
    speed_ranking = speed_score.sort_values()
    
    axes[1].barh(range(len(speed_ranking)), speed_ranking.values)
    axes[1].set_yticks(range(len(speed_ranking)))
    axes[1].set_yticklabels(speed_ranking.index)
    axes[1].set_xlabel('Speed Rank (lower is better)')
    axes[1].set_title('Model Speed Ranking')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return avg_metrics

def print_summary_report(df):
    # Print summary report of results
    if df is None or df.empty:
        print("No results to summarize")
        return
    
    print("\n" + "="*60)
    print("BENCHMARK REPORT")
    print("-"*60)
    
    # Overall statistics
    print(f"DOF Range Tested: {df['dof'].min()} to {df['dof'].max()}")
    print(f"Models Tested: {', '.join(df['model'].unique())}")
    print(f"Total Experiments: {len(df)}")
    
    # Best performers
    print(f"\n🏆 BEST PERFORMERS:")
    best_pos = df.loc[df['position_rmse'].idxmin()]
    best_joint = df.loc[df['joint_rmse'].idxmin()]
    fastest = df.loc[df['training_time'].idxmin()]
    
    print(f"  Best Position Accuracy: {best_pos['model']} (DOF={best_pos['dof']}, RMSE={best_pos['position_rmse']:.4f})")
    print(f"  Best Joint Accuracy: {best_joint['model']} (DOF={best_joint['dof']}, RMSE={best_joint['joint_rmse']:.4f})")
    print(f"  Fastest Training: {fastest['model']} (DOF={fastest['dof']}, Time={fastest['training_time']:.2f}s)")
    
    # Average performance across all DOFs
    print(f"\nAVERAGE PERFORMANCE (across all DOFs):")
    avg_perf = df.groupby('model')[['position_rmse', 'joint_rmse', 'training_time']].mean().round(4)
    print(avg_perf)
    
    print("="*60)

def create_models(input_dim=6, output_dim=3):
    # Create all available models
    models = {}
    
    # Always available models
    models['ANN'] = ANNModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=[64, 32], 
        epochs=20
    )
    models['KNN'] = KNNModel(n_neighbors=5)
    models['ELM'] = ELMModel(input_dim=input_dim, output_dim=output_dim, hidden_dim=50)
    models['RandomForest'] = RandomForestModel(n_estimators=25)
    
    try:
        models['SVM'] = SVMModel(kernel='rbf', C=1.0, epsilon=0.1)
    except Exception as e:
        print(f"SVM model not available: {e}")
    
    try:
        models['GPR'] = GPRModel(kernel_type='rbf')
    except Exception as e:
        print(f"GPR model not available: {e}")
    
    try:
        models['MDN'] = MDNModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=[64, 32],
            n_mixtures=3,
            epochs=50
        )
    except Exception as e:
        print(f"MDN model not available: {e}")
    
    try:
        models['CVAE'] = CVAEModel(
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=8,
            epochs=50
        )
    except Exception as e:
        print(f"CVAE model not available: {e}")
    
    return models

def single_test(dof, models, data_path, results_path, sample_limit=None):
    # Test a single DOF configuration
    print(f"Testing DOF={dof}...")
    
    # Load data using updated format
    train_poses = data_path / 'Training' / f'{dof}_training.json'
    train_solutions = data_path / 'Training' / f'{dof}_training_solutions.json'
    test_poses = data_path / 'Testing' / f'{dof}_testing.json' 
    test_solutions = data_path / 'Testing' / f'{dof}_testing_solutions.json'
    
    # Check if files exist
    for file_path in [train_poses, train_solutions, test_poses, test_solutions]:
        if not file_path.exists():
            raise FileNotFoundError(f"Required data file not found: {file_path}")
    
    X_train, y_train = load_ik_data(train_poses, train_solutions)
    X_test, y_test = load_ik_data(test_poses, test_solutions)
    
    # Subsample if needed
    if sample_limit:
        if len(X_train) > sample_limit:
            idx = np.random.choice(len(X_train), sample_limit, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
        if len(X_test) > sample_limit//2:
            idx = np.random.choice(len(X_test), sample_limit//2, replace=False)
            X_test, y_test = X_test[idx], y_test[idx]
    
    print(f"  Data: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    print(f"  Input dims: {X_train.shape[1]} (pose), Output dims: {y_train.shape[1]} (joints)")
    
    # Update model dimensions based on actual data
    for model in models.values():
        if hasattr(model, 'model_params'):
            model.model_params['input_dim'] = X_train.shape[1]
            model.model_params['output_dim'] = y_train.shape[1]
        
        if hasattr(model, 'input_dim'):
            model.input_dim = X_train.shape[1]
        if hasattr(model, 'output_dim'):
            model.output_dim = y_train.shape[1]
    
    # Train and evaluate
    training_results = train_all_models(models, X_train, y_train)
    evaluation_results = evaluate_all_models(training_results, X_test, y_test)
    df = create_results_dataframe(evaluation_results)
    
    if not df.empty:
        df['dof'] = dof
        results_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path / f'dof_{dof}_results.csv', index=False)
        print(f"  ✓ Results saved for DOF={dof}")
    
    return df

def multiple_test(dof_range, models, data_path, results_path, sample_limit=None):
    # Test multiple DOF configurations
    all_results = []
    
    for dof in dof_range:
        try:
            df = single_test(dof, models, data_path, results_path, sample_limit)
            if not df.empty:
                all_results.append(df)
        except Exception as e:
            print(f"  ✗ Failed DOF={dof}: {e}")
    
    if not all_results:
        print("No successful results to combine")
        return None
    
    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(results_path / 'all_results.csv', index=False)
    
    return combined_df