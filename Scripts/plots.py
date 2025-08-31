import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def setup_plot_style():
    # Setup consistent visual style for all plots
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'gray'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['font.size'] = 10

def get_model_colors(results_df):
    # Create consistent color mapping for models using matplotlib colors
    unique_models = results_df['model'].unique()
    # Use matplotlib's tab colors and cycle through them
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
    return dict(zip(unique_models, colors))

def plot_accuracy_vs_speed_tradeoff(results_df, ax, model_colors):
    # Plot accuracy vs speed tradeoff for ML models
    ml_df = results_df[results_df['model_type'] == 'ML'] if 'model_type' in results_df.columns else results_df
    
    for model in ml_df['model'].unique():
        model_data = ml_df[ml_df['model'] == model]
        avg_acc = model_data['joint_rmse'].mean()
        avg_time = model_data['training_time'].mean()
        ax.scatter(avg_time, avg_acc, s=120, alpha=0.8,
                  label=model, color=model_colors.get(model, 'blue'), 
                  edgecolors='white', linewidth=1)
        ax.annotate(model, (avg_time, avg_acc), xytext=(8, 8),
                   textcoords='offset points', fontsize=9, fontweight='bold')

    ax.set_xlabel('Average Training Time (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Joint RMSE', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy vs Speed Tradeoff', fontsize=12, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

def plot_performance_heatmap(results_df, ax):
    # Plot model performance heatmap
    ml_df = results_df[results_df['model_type'] == 'ML'] if 'model_type' in results_df.columns else results_df
    pivot_rmse = ml_df.pivot_table(values='joint_rmse', index='model', columns='dof', aggfunc='mean')

    if not pivot_rmse.empty:
        im = ax.imshow(pivot_rmse.values, cmap='viridis', aspect='auto', interpolation='bilinear')
        ax.set_xticks(range(len(pivot_rmse.columns)))
        ax.set_xticklabels(pivot_rmse.columns, fontweight='bold')
        ax.set_yticks(range(len(pivot_rmse.index)))
        ax.set_yticklabels(pivot_rmse.index, fontweight='bold')
        ax.set_xlabel('DOF', fontsize=11, fontweight='bold')
        ax.set_ylabel('Model', fontsize=11, fontweight='bold')
        ax.set_title('Joint RMSE Heatmap', fontsize=12, fontweight='bold', pad=20)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Joint RMSE', fontweight='bold')

def plot_training_time_distribution(results_df, ax):
    # Plot training time distribution
    ml_df = results_df[results_df['model_type'] == 'ML'] if 'model_type' in results_df.columns else results_df
    training_times = ml_df.groupby('model')['training_time'].mean().sort_values()
    colors = ['green' if t < 10 else 'orange' if t < 30 else 'red' for t in training_times.values]

    bars = ax.bar(range(len(training_times)), training_times.values, 
                  color=colors, alpha=0.8, edgecolor='white')
    ax.set_xticks(range(len(training_times)))
    ax.set_xticklabels(training_times.index, rotation=45, ha='right', fontweight='bold')
    ax.set_ylabel('Training Time (s)', fontsize=11, fontweight='bold')
    ax.set_title('Average Training Time by Model', fontsize=12, fontweight='bold', pad=20)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, training_times.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{val:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

def plot_inference_speed_comparison(results_df, ax):
    # Plot inference speed comparisons
    ml_df = results_df[results_df['model_type'] == 'ML'] if 'model_type' in results_df.columns else results_df
    inference_times = ml_df.groupby('model')['inference_time_per_sample'].mean() * 1000
    inference_times = inference_times.sort_values()
    colors = ['green' if t < 1 else 'orange' if t < 5 else 'red' for t in inference_times.values]

    bars = ax.barh(range(len(inference_times)), inference_times.values, 
                   color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(inference_times)))
    ax.set_yticklabels(inference_times.index, fontweight='bold')
    ax.set_xlabel('Inference Time (ms/sample)', fontsize=11, fontweight='bold')
    ax.set_title('Inference Speed Ranking', fontsize=12, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

def plot_best_model_distribution(results_df, ax):
    # Plot pie chart of best model distribution
    ml_df = results_df[results_df['model_type'] == 'ML'] if 'model_type' in results_df.columns else results_df
    
    if 'dof' in ml_df.columns and len(ml_df) > 0:
        best_per_dof = ml_df.loc[ml_df.groupby('dof')['joint_rmse'].idxmin()]
        dof_counts = best_per_dof['model'].value_counts()

        colors = plt.cm.tab10(np.linspace(0, 1, len(dof_counts)))
        wedges, texts, autotexts = ax.pie(dof_counts.values, labels=dof_counts.index, 
                                         autopct='%1.1f%%', startangle=90, colors=colors, 
                                         textprops={'fontweight': 'bold'})
        ax.set_title('Best Model Distribution Across DOFs', fontsize=12, fontweight='bold', pad=20)

def plot_performance_vs_dof(results_df, ax):
    # Plot performance improvement with increasing DOF
    ml_df = results_df[results_df['model_type'] == 'ML'] if 'model_type' in results_df.columns else results_df
    
    for model in ml_df['model'].unique():
        model_data = ml_df[ml_df['model'] == model].sort_values('dof')
        if len(model_data) > 1:
            improvement = (model_data['joint_rmse'].values[:-1] - model_data['joint_rmse'].values[1:]) / model_data['joint_rmse'].values[:-1] * 100
            ax.plot(model_data['dof'].values[1:], improvement, 'o-', label=model, alpha=0.8, linewidth=2)

    ax.set_xlabel('DOF', fontsize=11, fontweight='bold')
    ax.set_ylabel('Performance Change (%)', fontsize=11, fontweight='bold')
    ax.set_title('Performance Change with Increasing DOF', fontsize=12, fontweight='bold', pad=20)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

def plot_traditional_vs_ml_speed_comparison(results_df):
    # Create two plots comparing Traditional vs ML inference speeds across DOF
    if 'model_type' not in results_df.columns or 'dof' not in results_df.columns:
        print("Warning: Missing model_type or dof columns. Cannot create traditional vs ML comparison.")
        return None
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Get unique DOF values
    dof_values = sorted(results_df['dof'].unique())
    
    if len(dof_values) == 0:
        print("Warning: No DOF data found.")
        return None
    
    # Define timeout threshold (in ms) - adjust based on your timeout settings
    timeout_threshold = 1000  # 1000ms = 1s, adjust based on your timeout settings
    
    # Plot 1: Average ML vs Average Traditional by DOF
    ml_avg_times = []
    trad_avg_times = []
    
    for dof in dof_values:
        dof_data = results_df[results_df['dof'] == dof]
        
        # Calculate average inference time for ML models at this DOF
        ml_data = dof_data[dof_data['model_type'] == 'ML']
        ml_avg = ml_data['inference_time_per_sample'].mean() * 1000 if len(ml_data) > 0 else 0
        ml_avg_times.append(ml_avg)
        
        # Calculate average inference time for Traditional models at this DOF
        trad_data = dof_data[dof_data['model_type'] == 'Traditional']
        trad_avg = trad_data['inference_time_per_sample'].mean() * 1000 if len(trad_data) > 0 else 0
        trad_avg_times.append(trad_avg)
    
    # Plot averaged lines
    ax1.plot(dof_values, ml_avg_times, 'o-', color='blue', linewidth=3, markersize=8, 
             label='ML Models (Average)', alpha=0.9)
    ax1.plot(dof_values, trad_avg_times, 's-', color='red', linewidth=3, markersize=8, 
             label='Traditional IK (Average)', alpha=0.9)
    
    # Add timeout region
    ax1.axhspan(timeout_threshold, ax1.get_ylim()[1] if max(max(ml_avg_times), max(trad_avg_times)) > timeout_threshold 
                else timeout_threshold * 2, alpha=0.3, color='red', label='Timeout Error Region')
    
    ax1.set_xlabel('DOF', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inference Time (ms per sample)', fontsize=12, fontweight='bold')
    ax1.set_title('Average Solve Time: Traditional vs ML Methods', fontsize=14, fontweight='bold', pad=20)
    ax1.set_yscale('log')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(dof_values)
    
    # Add speedup annotation
    if len(ml_avg_times) > 0 and len(trad_avg_times) > 0:
        avg_speedup = np.mean([t/m for t, m in zip(trad_avg_times, ml_avg_times) if m > 0])
        ax1.text(0.02, 0.98, f'Traditional methods are\n{avg_speedup:.1f}x slower on average', 
                 transform=ax1.transAxes, ha='left', va='top', fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: All Individual Models by DOF
    ml_models = results_df[results_df['model_type'] == 'ML']['model'].unique()
    trad_models = results_df[results_df['model_type'] == 'Traditional']['model'].unique()
    
    # Define colors
    ml_colors = plt.cm.Blues(np.linspace(0.4, 1, len(ml_models)))
    trad_colors = plt.cm.Reds(np.linspace(0.4, 1, len(trad_models)))
    
    # Plot ML models
    for i, model in enumerate(ml_models):
        model_data = results_df[results_df['model'] == model]
        if len(model_data) > 0:
            dofs = model_data['dof'].values
            times = model_data['inference_time_per_sample'].values * 1000
            ax2.plot(dofs, times, 'o-', color=ml_colors[i], linewidth=2, markersize=6, 
                     label=f'{model} (ML)', alpha=0.8)
    
    # Plot Traditional models
    for i, model in enumerate(trad_models):
        model_data = results_df[results_df['model'] == model]
        if len(model_data) > 0:
            dofs = model_data['dof'].values
            times = model_data['inference_time_per_sample'].values * 1000
            ax2.plot(dofs, times, 's-', color=trad_colors[i], linewidth=2, markersize=6, 
                     label=f'{model} (Traditional)', alpha=0.8)
    
    # Add timeout region
    y_max = results_df['inference_time_per_sample'].max() * 1000
    timeout_upper = max(timeout_threshold * 2, y_max * 1.2)
    ax2.axhspan(timeout_threshold, timeout_upper, alpha=0.3, color='red', label='Timeout Error Region')
    
    ax2.set_xlabel('DOF', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Inference Time (ms per sample)', fontsize=12, fontweight='bold')
    ax2.set_title('Individual Model Performance Across DOF', fontsize=14, fontweight='bold', pad=20)
    ax2.set_yscale('log')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(dof_values)
    
    # Add timeout threshold line
    ax1.axhline(y=timeout_threshold, color='darkred', linestyle='--', alpha=0.8, linewidth=2)
    ax2.axhline(y=timeout_threshold, color='darkred', linestyle='--', alpha=0.8, linewidth=2)
    
    # Add text annotation for timeout line
    ax1.text(min(dof_values), timeout_threshold * 1.1, 'Timeout Threshold', 
             fontsize=10, fontweight='bold', color='darkred')
    ax2.text(min(dof_values), timeout_threshold * 1.1, 'Timeout Threshold', 
             fontsize=10, fontweight='bold', color='darkred')
    
    plt.tight_layout()
    return fig

def plot_3d_tradeoff_analysis(results_df, model_colors):
    # 3D plot showing DOF vs Inference Speed vs Joint RMSE
    ml_df = results_df[results_df['model_type'] == 'ML'] if 'model_type' in results_df.columns else results_df
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    for model in ml_df['model'].unique():
        model_data = ml_df[ml_df['model'] == model]
        x = model_data['dof']
        y = model_data['inference_time_per_sample'] * 1000  # Convert to ms
        z = model_data['joint_rmse']

        ax.scatter(x, y, z, c=[model_colors.get(model, 'blue')], s=80, alpha=0.8,
                  label=model, edgecolors='white', linewidth=1)

    ax.set_xlabel('DOF', fontsize=11, fontweight='bold')
    ax.set_ylabel('Inference Time (ms)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Joint RMSE', fontsize=11, fontweight='bold')
    ax.set_title('3D Model Performance Tradeoffs\n(DOF vs Speed vs Accuracy)',
                fontsize=14, fontweight='bold', pad=20)

    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    plt.tight_layout()
    return fig

def plot_pareto_frontier_analysis(results_df, model_colors):
    # Create Pareto frontier analysis for accuracy vs inference speed
    ml_df = results_df[results_df['model_type'] == 'ML'] if 'model_type' in results_df.columns else results_df
    
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate average metrics per model
    model_metrics = ml_df.groupby('model').agg({
        'joint_rmse': 'mean',
        'inference_time_per_sample': 'mean'
    }).reset_index()

    # Convert to ms
    model_metrics['inference_time_ms'] = model_metrics['inference_time_per_sample'] * 1000

    # Plot all models
    for _, row in model_metrics.iterrows():
        ax.scatter(row['inference_time_ms'], row['joint_rmse'],
                  s=150, alpha=0.8, color=model_colors.get(row['model'], 'blue'),
                  edgecolors='white', linewidth=2, label=row['model'])
        ax.annotate(row['model'], (row['inference_time_ms'], row['joint_rmse']),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, fontweight='bold')

    # Find Pareto frontier (minimize both inference time and RMSE)
    points = model_metrics[['inference_time_ms', 'joint_rmse']].values
    pareto_front = []

    for i, point in enumerate(points):
        dominated = False
        for other_point in points:
            if (other_point[0] <= point[0] and other_point[1] <= point[1] and
                (other_point[0] < point[0] or other_point[1] < point[1])):
                dominated = True
                break
        if not dominated:
            pareto_front.append(i)

    # Plot Pareto frontier
    if len(pareto_front) > 1:
        pareto_points = points[pareto_front]
        sorted_indices = np.argsort(pareto_points[:, 0])
        pareto_sorted = pareto_points[sorted_indices]
        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], 'r--',
               linewidth=3, alpha=0.8, label='Pareto Frontier')

    ax.set_xlabel('Inference Time (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Joint RMSE (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Frontier Analysis: Accuracy vs Speed Tradeoff',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def plot_variance_analysis(results_df, model_colors):
    # Create error bar plots showing model stability across DOF
    ml_df = results_df[results_df['model_type'] == 'ML'] if 'model_type' in results_df.columns else results_df
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Joint RMSE variance across DOF
    for model in ml_df['model'].unique():
        model_data = ml_df[ml_df['model'] == model]
        if len(model_data) > 0 and 'dof' in model_data.columns:
            dof_stats = model_data.groupby('dof')['joint_rmse'].agg(['mean', 'std']).reset_index()

            ax1.errorbar(dof_stats['dof'], dof_stats['mean'], yerr=dof_stats['std'],
                        marker='o', linewidth=2, markersize=6, alpha=0.8,
                        color=model_colors.get(model, 'blue'), label=model, capsize=5)

    ax1.set_xlabel('DOF', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Joint RMSE', fontsize=12, fontweight='bold')
    ax1.set_title('Model Stability: RMSE vs DOF', fontsize=12, fontweight='bold', pad=20)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Training time variance across DOF
    for model in ml_df['model'].unique():
        model_data = ml_df[ml_df['model'] == model]
        if len(model_data) > 0 and 'dof' in model_data.columns:
            dof_stats = model_data.groupby('dof')['training_time'].agg(['mean', 'std']).reset_index()

            ax2.errorbar(dof_stats['dof'], dof_stats['mean'], yerr=dof_stats['std'],
                        marker='s', linewidth=2, markersize=6, alpha=0.8,
                        color=model_colors.get(model, 'blue'), label=model, capsize=5)

    ax2.set_xlabel('DOF', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Time (s)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Stability: Training Time vs DOF', fontsize=12, fontweight='bold', pad=20)
    ax2.set_yscale('log')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def create_comprehensive_dashboard(results_df, save_path=None):
    # Create the main comprehensive analysis dashboard
    setup_plot_style()
    
    # Filter for ML models for consistency with original plots
    ml_results_df = results_df[results_df['model_type'] == 'ML'] if 'model_type' in results_df.columns else results_df
    model_colors = get_model_colors(ml_results_df)

    # Main dashboard
    fig_main = plt.figure(figsize=(20, 12))
    gs = fig_main.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
    fig_main.suptitle('Model Performance Analysis Dashboard', fontsize=16, fontweight='bold', y=0.95)

    # Create all subplots
    ax1 = fig_main.add_subplot(gs[0, 0])
    plot_accuracy_vs_speed_tradeoff(ml_results_df, ax1, model_colors)

    ax2 = fig_main.add_subplot(gs[0, 1])
    plot_performance_heatmap(ml_results_df, ax2)

    ax3 = fig_main.add_subplot(gs[0, 2])
    plot_training_time_distribution(ml_results_df, ax3)

    ax4 = fig_main.add_subplot(gs[1, 0])
    plot_inference_speed_comparison(ml_results_df, ax4)

    ax5 = fig_main.add_subplot(gs[1, 1])
    plot_best_model_distribution(ml_results_df, ax5)

    ax6 = fig_main.add_subplot(gs[1, 2])
    plot_performance_vs_dof(ml_results_df, ax6)

    if save_path:
        plt.savefig(save_path / 'comprehensive_analysis.png', dpi=150, bbox_inches='tight')
    
    return fig_main

def generate_all_visualizations(results_df, save_path=None):
    # Generate all visualization plots and return them as a dictionary
    setup_plot_style()
    
    plots = {}
    model_colors = get_model_colors(results_df)
    
    # Main comprehensive dashboard
    plots['main_dashboard'] = create_comprehensive_dashboard(results_df, save_path)
    
    # Traditional vs ML comparison (only if traditional data exists)
    if 'model_type' in results_df.columns:
        plots['traditional_vs_ml'] = plot_traditional_vs_ml_speed_comparison(results_df)
        if save_path and plots['traditional_vs_ml']:
            plots['traditional_vs_ml'].savefig(save_path / 'traditional_vs_ml_comparison.png', 
                                             dpi=150, bbox_inches='tight')
    
    # Advanced visualizations
    plots['3d_tradeoff'] = plot_3d_tradeoff_analysis(results_df, model_colors)
    if save_path:
        plots['3d_tradeoff'].savefig(save_path / '3d_tradeoff_analysis.png', dpi=150, bbox_inches='tight')
    
    plots['pareto_frontier'] = plot_pareto_frontier_analysis(results_df, model_colors)
    if save_path:
        plots['pareto_frontier'].savefig(save_path / 'pareto_frontier_analysis.png', dpi=150, bbox_inches='tight')
    
    plots['variance_analysis'] = plot_variance_analysis(results_df, model_colors)
    if save_path:
        plots['variance_analysis'].savefig(save_path / 'variance_analysis.png', dpi=150, bbox_inches='tight')
    
    return plots