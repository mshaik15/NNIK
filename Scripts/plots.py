import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')


def style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.15,
        'grid.color': '#666666',
        'lines.linewidth': 2,
        'lines.markersize': 7
    })

def colors():
    # Get color palette based on inferno colormap
    return {
        'ml': '#FCA311',           # Warm orange/yellow
        'traditional': '#370617',   # Deep purple
        'accent': '#DC2F02',        # Red accent
        'highlight': '#FFBA08'      # Bright yellow
    }

def model_colors(models):
    # Generate colors for individual models using gradient
    n = len(models)
    cmap = plt.cm.YlOrRd
    return {model: cmap(i/n) for i, model in enumerate(models)}


def split_by_type(df):
    # Split dataframe by model type
    ml_df = df[df['model_type'] == 'ML'] if 'model_type' in df.columns else df
    trad_df = df[df['model_type'] == 'Traditional'] if 'model_type' in df.columns else pd.DataFrame()
    return ml_df, trad_df

def calculate_stats(df, group_col, value_col, scale=1):
    # Calculate mean and std for grouped data
    grouped = df.groupby(group_col)[value_col]
    means = grouped.mean() * scale
    stds = grouped.std() * scale
    return means, stds


def plot_accuracy_speed_tradeoff(df, save_path=None):
    # Accuracy vs Speed Tradeoff
    style()
    ml_df, _ = split_by_type(df)
    model_color_map = model_colors(ml_df['model'].unique())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model in ml_df['model'].unique():
        data = ml_df[ml_df['model'] == model]
        ax.scatter(data['training_time'].mean(), data['joint_rmse'].mean(),
                  s=120, alpha=0.8, color=model_color_map[model], 
                  edgecolors='white', linewidth=2, label=model)
    
    ax.set_xlabel('Training Time (s)', fontweight='bold')
    ax.set_ylabel('Joint RMSE', fontweight='bold')
    ax.set_title('Accuracy-Speed Tradeoff', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    return fig

def plot_rmse_heatmap(df, save_path=None):
    # RMSE Heatmap across models and DOF
    style()
    ml_df, _ = split_by_type(df)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    pivot = ml_df.pivot_table(values='joint_rmse', index='model', columns='dof', aggfunc='mean')
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', interpolation='bilinear')
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('DOF', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    ax.set_title('Joint RMSE Heatmap', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='RMSE')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    return fig

def plot_bar_comparison(df, metric, title, xlabel, log_scale=False, save_path=None):
    # Generic horizontal bar chart for model comparisons
    style()
    ml_df, _ = split_by_type(df)
    model_color_map = model_colors(ml_df['model'].unique())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    values = ml_df.groupby('model')[metric].mean().sort_values()
    
    bars = ax.barh(range(len(values)), values.values,
                   color=[model_color_map[m] for m in values.index],
                   alpha=0.8, edgecolor='white', linewidth=2)
    
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(values.index)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if log_scale:
        ax.set_xscale('log')
    
    # Add value labels
    for bar, val in zip(bars, values.values):
        unit = 's' if 'time' in metric.lower() else 'ms' if 'inference' in metric.lower() else ''
        ax.text(val, bar.get_y() + bar.get_height()/2,
               f' {val:.2f}{unit}', va='center', fontsize=9)
    
    ax.grid(True, alpha=0.2, axis='x')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    return fig

def plot_line_comparison(df, x_col, y_col, title, ylabel, save_path=None):
    # Generic line plot for model comparisons
    style()
    ml_df, _ = split_by_type(df)
    model_color_map = model_colors(ml_df['model'].unique())
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    for model in ml_df['model'].unique():
        data = ml_df[ml_df['model'] == model].groupby(x_col)[y_col].mean()
        ax.plot(data.index, data.values, 'o-', color=model_color_map[model],
               label=model, linewidth=2.5, markersize=7, alpha=0.8)
    
    ax.set_xlabel('Degrees of Freedom', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    return fig

def plot_ml_vs_traditional(df, metric, title, ylabel, use_bars=True, save_path=None):
    # Compare ML vs Traditional methods
    style()
    color_palette = colors()
    ml_df, trad_df = split_by_type(df)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    scale = 1000 if 'inference' in metric else 1
    ml_means, _ = calculate_stats(ml_df, 'dof', metric, scale)
    trad_means, _ = calculate_stats(trad_df, 'dof', metric, scale)
    
    x = np.array(ml_means.index)
    
    if use_bars:
        width = 0.35
        ax.bar(x - width/2, ml_means.values, width, label='ML Methods',
               color=color_palette['ml'], alpha=0.8, edgecolor='white', linewidth=2)
        ax.bar(x + width/2, trad_means.values, width, label='Traditional',
               color=color_palette['traditional'], alpha=0.8, edgecolor='white', linewidth=2)
        ax.set_xticks(x)
    else:
        ax.plot(x, ml_means.values, 'o-', color=color_palette['ml'], 
                linewidth=3, markersize=8, label='ML Methods', alpha=0.9)
        ax.plot(x, trad_means.values, 's-', color=color_palette['traditional'],
                linewidth=3, markersize=8, label='Traditional', alpha=0.9)
        
        if 'inference' in metric:
            ax.set_yscale('log')
            speedup = (trad_means.values / ml_means.values).mean()
            ax.text(0.98, 0.02, f'ML is {speedup:.1f}× faster',
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor=color_palette['highlight'], alpha=0.3),
                   fontweight='bold')
    
    ax.set_xlabel('Degrees of Freedom', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    return fig

def plot_speedup_factors(df, save_path=None):
    # Plot speedup factors of ML over traditional
    style()
    color_palette = colors()
    ml_df, trad_df = split_by_type(df)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    ml_means, _ = calculate_stats(ml_df, 'dof', 'inference_time_per_sample')
    trad_means, _ = calculate_stats(trad_df, 'dof', 'inference_time_per_sample')
    
    speedup = trad_means / ml_means
    x = speedup.index
    
    bars = ax.bar(x, speedup.values, 
                 color=plt.cm.YlOrRd(speedup.values / speedup.max()),
                 alpha=0.8, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars, speedup.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{val:.1f}×', ha='center', va='bottom', fontweight='bold')
    
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Degrees of Freedom', fontweight='bold')
    ax.set_ylabel('Speedup Factor (Traditional/ML)', fontweight='bold')
    ax.set_title('ML Speedup Over Traditional Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.2, axis='y')
    
    # Add summary
    ax.text(0.98, 0.98, f'Avg: {speedup.mean():.1f}×\nMax: {speedup.max():.1f}×',
           transform=ax.transAxes, ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor=color_palette['highlight'], alpha=0.3),
           fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    return fig

def create_comprehensive_dashboard(df, save_path=None):
    """6-panel ML model analysis dashboard"""
    style()
    ml_df, _ = split_by_type(df)
    model_color_map = model_colors(ml_df['model'].unique())
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.4)
    
    # Plot 1: Accuracy-Speed Tradeoff (spans 2 columns)
    ax = fig.add_subplot(gs[0, :2])
    for model in ml_df['model'].unique():
        data = ml_df[ml_df['model'] == model]
        ax.scatter(data['training_time'].mean(), data['joint_rmse'].mean(),
                  s=100, alpha=0.8, color=model_color_map[model], 
                  edgecolors='white', linewidth=1.5, label=model)
    ax.set_xlabel('Training Time (s)', fontweight='bold')
    ax.set_ylabel('Joint RMSE', fontweight='bold')
    ax.set_title('Accuracy-Speed Tradeoff')
    ax.set_xscale('log')
    ax.legend(loc='best', fontsize=8)
    
    # Plot 2: RMSE Heatmap
    ax = fig.add_subplot(gs[0, 2])
    pivot = ml_df.pivot_table(values='joint_rmse', index='model', columns='dof', aggfunc='mean')
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel('DOF', fontweight='bold')
    ax.set_title('RMSE Heatmap')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Plot 3: Training Time
    ax = fig.add_subplot(gs[1, 0])
    times = ml_df.groupby('model')['training_time'].mean().sort_values()
    ax.barh(range(len(times)), times.values,
           color=[model_color_map[m] for m in times.index], alpha=0.8)
    ax.set_yticks(range(len(times)))
    ax.set_yticklabels(times.index, fontsize=8)
    ax.set_xlabel('Training Time (s)', fontweight='bold')
    ax.set_title('Training Duration')
    ax.set_xscale('log')
    
    # Plot 4: Inference Speed
    ax = fig.add_subplot(gs[1, 1])
    inference = ml_df.groupby('model')['inference_time_per_sample'].mean() * 1000
    inference = inference.sort_values()
    ax.barh(range(len(inference)), inference.values,
           color=[model_color_map[m] for m in inference.index], alpha=0.8)
    ax.set_yticks(range(len(inference)))
    ax.set_yticklabels(inference.index, fontsize=8)
    ax.set_xlabel('Inference (ms)', fontweight='bold')
    ax.set_title('Inference Speed')
    ax.set_xscale('log')
    
    # Plot 5: Performance vs DOF
    ax = fig.add_subplot(gs[1, 2])
    for model in ml_df['model'].unique():
        data = ml_df[ml_df['model'] == model].groupby('dof')['joint_rmse'].mean()
        ax.plot(data.index, data.values, 'o-', color=model_color_map[model],
               label=model, linewidth=2, markersize=5, alpha=0.8)
    ax.set_xlabel('DOF', fontweight='bold')
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('Accuracy vs DOF')
    ax.legend(loc='best', fontsize=7)
    
    # Plot 6: Summary Table
    ax = fig.add_subplot(gs[2, :])
    ax.axis('tight')
    ax.axis('off')
    
    summary = [[model, 
                f"{ml_df[ml_df['model']==model]['joint_rmse'].mean():.4f}",
                f"{ml_df[ml_df['model']==model]['training_time'].mean():.2f}",
                f"{ml_df[ml_df['model']==model]['inference_time_per_sample'].mean()*1000:.2f}"]
               for model in ml_df['model'].unique()]
    
    table = ax.table(cellText=summary,
                    colLabels=['Model', 'Avg RMSE', 'Train (s)', 'Inference (ms)'],
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    fig.suptitle('Machine Learning Model Performance Analysis', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    return fig

def generate_all_figures(df, output_dir='./figures/'):
    # Generate all figure
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating Academic Research Figures")
    print("=" * 60)
    
    # Individual plot
    plots = [
        ("Accuracy-Speed Tradeoff", 
         lambda: plot_accuracy_speed_tradeoff(df, f"{output_dir}/accuracy_speed_tradeoff.pdf")),
        ("RMSE Heatmap", 
         lambda: plot_rmse_heatmap(df, f"{output_dir}/rmse_heatmap.pdf")),
        ("Training Time Distribution", 
         lambda: plot_bar_comparison(df, 'training_time', 'Training Duration', 'Time (s)', 
                                    True, f"{output_dir}/training_time.pdf")),
        ("Inference Speed Ranking", 
         lambda: plot_bar_comparison(df, 'inference_time_per_sample', 'Inference Speed', 
                                    'Time (ms)', True, f"{output_dir}/inference_speed.pdf")),
        ("Accuracy vs DOF", 
         lambda: plot_line_comparison(df, 'dof', 'joint_rmse', 'Accuracy vs DOF', 
                                     'Joint RMSE', f"{output_dir}/accuracy_vs_dof.pdf")),
        ("ML vs Traditional Accuracy", 
         lambda: plot_ml_vs_traditional(df, 'joint_rmse', 'Accuracy Comparison', 
                                       'Joint RMSE', True, f"{output_dir}/ml_vs_trad_accuracy.pdf")),
        ("ML vs Traditional Speed", 
         lambda: plot_ml_vs_traditional(df, 'inference_time_per_sample', 'Speed Comparison', 
                                       'Inference Time (ms)', False, f"{output_dir}/ml_vs_trad_speed.pdf")),
        ("Speedup Factors", 
         lambda: plot_speedup_factors(df, f"{output_dir}/speedup_factors.pdf")),
        ("Comprehensive Dashboard", 
         lambda: create_comprehensive_dashboard(df, f"{output_dir}/comprehensive_analysis.pdf"))
    ]
    
    for name, func in plots:
        print(f"  ✓ {name}")
        func()
    
    print(f"\n✅ Generated {len(plots)} PDF files in {output_dir}")
    print("\nAll figures use:")
    print("  • Professional academic style")
    print("  • Inferno-inspired color scheme")
    print("  • 300 DPI resolution")
    
    return True

if __name__ == "__main__":
    print("Academic figure generator ready!")
    print("Usage: generate_all_figures(your_dataframe, './output_dir/')")
    print("\nExpected DataFrame columns:")
    print("  - model, model_type, dof, joint_rmse")
    print("  - training_time, inference_time_per_sample")