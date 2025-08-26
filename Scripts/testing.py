import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from training import load_ik_data, train_all_models, evaluate_all_models, create_results_dataframe

def run_single_test(dof, models, data_path, results_path, sample_limit=None):
    """Run test for single DOF configuration"""
    print(f"Testing DOF={dof}...")
    
    # Load data
    train_poses = data_path / 'Training' / f'{dof}_training.json'
    train_solutions = data_path / 'Training' / f'{dof}_training_solutions.json'
    test_poses = data_path / 'Testing' / f'{dof}_testing.json'
    test_solutions = data_path / 'Testing' / f'{dof}_testing_solutions.json'
    
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
    
    # Update model dimensions
    for model in models.values():
        if hasattr(model, 'model_params'):
            model.model_params['input_dim'] = X_train.shape[1]
            model.model_params['output_dim'] = y_train.shape[1]
    
    # Train and evaluate
    training_results = train_all_models(models, X_train, y_train)
    evaluation_results = evaluate_all_models(training_results, X_test, y_test)
    df = create_results_dataframe(evaluation_results)
    
    if not df.empty:
        df['dof'] = dof
        # Save individual results
        df.to_csv(results_path / f'dof_{dof}_results.csv', index=False)
        print(f"  ✓ Results saved for DOF={dof}")
    
    return df

def run_multi_dof_test(dof_range, models, data_path, results_path, sample_limit=None):
    """Run tests across multiple DOF configurations"""
    all_results = []
    
    for dof in dof_range:
        try:
            df = run_single_test(dof, models, data_path, results_path, sample_limit)
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

def plot_scalability_analysis(df, save_path=None):
    """Create time vs DOF and RMSE vs DOF plots"""
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
    """Show which models are best at different metrics"""
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
    """Print a clean summary report"""
    if df is None or df.empty:
        print("No results to summarize")
        return
    
    print("\n" + "="*60)
    print("📊 BENCHMARKING SUMMARY REPORT")
    print("="*60)
    
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
    print(f"\n📈 AVERAGE PERFORMANCE (across all DOFs):")
    avg_perf = df.groupby('model')[['position_rmse', 'joint_rmse', 'training_time']].mean().round(4)
    print(avg_perf)
    
    print("="*60)

def quick_test(dof_list=[3, 4, 5], model_list=['ANN', 'KNN', 'ELM'], sample_limit=500):
    """Quick test function for development"""
    from pathlib import Path
    
    # Paths (adjust these)
    PROJECT_PATH = Path('.')  # Adjust to your project path
    DATA_PATH = PROJECT_PATH / 'data'
    RESULTS_PATH = PROJECT_PATH / 'results'
    RESULTS_PATH.mkdir(exist_ok=True)
    
    # Import models
    from Models.ann import ANNModel
    from Models.knn import KNNModel
    from Models.elm import ELMModel
    from Models.random_forest import RandomForestModel
    
    # Create models
    models = {}
    if 'ANN' in model_list:
        models['ANN'] = ANNModel(hidden_layers=[64, 32], epochs=20)
    if 'KNN' in model_list:
        models['KNN'] = KNNModel(n_neighbors=5)
    if 'ELM' in model_list:
        models['ELM'] = ELMModel(hidden_dim=50)
    if 'Random_Forest' in model_list:
        models['Random_Forest'] = RandomForestModel(n_estimators=25)
    
    print(f"🚀 Quick test: DOFs={dof_list}, Models={list(models.keys())}")
    
    # Run tests
    results_df = run_multi_dof_test(dof_list, models, DATA_PATH, RESULTS_PATH, sample_limit)
    
    if results_df is not None:
        # Generate plots and summary
        plot_scalability_analysis(results_df, RESULTS_PATH / 'scalability.png')
        avg_metrics = plot_model_ranking(results_df, RESULTS_PATH / 'ranking.png')
        print_summary_report(results_df)
        
        return results_df, avg_metrics
    
    return None, None

if __name__ == "__main__":
    # Run quick test
    results, metrics = quick_test()