import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Tuple
import torch
from concurrent.futures import ThreadPoolExecutor

def load_ik_data(poses_file: Path, solutions_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    # Load inverse kinematics data from JSON file
    with open(poses_file, 'r') as f:
        poses_data = json.load(f)
    with open(solutions_file, 'r') as f:
        solutions_data = json.load(f)

    X = np.array([item['pose'] for item in poses_data['data']], dtype=np.float32)
    y = np.array([item['joint_angles'] for item in solutions_data['data']], dtype=np.float32)
    
    print(f"Loaded data: X shape = {X.shape}, y shape = {y.shape}")
    print(f"DOF from data: {y.shape[1]}")
    
    return X, y

def setup_gpu() -> torch.device:
    # Setup GPU if available and return device
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU")
    return device

def train_single_model(args) -> Dict:
    # Train a single model with GPU support
    name, model, X_train, y_train, device = args
    
    print(f"Training {name}...")
    
    try:
        # Check if model should use GPU
        use_gpu = name in ['ANN', 'MDN', 'CVAE'] and device.type == 'cuda'
        
        if use_gpu:
            if hasattr(model, 'device'):
                model.device = device
            elif hasattr(model, 'to'):
                model.to(device)
            print(f"  Using GPU for {name}")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Clear GPU cache if used
        if use_gpu:
            torch.cuda.empty_cache()
        
        print(f"  ✓ {name} completed in {training_time:.2f}s")
        
        return {
            'name': name,
            'model': model,
            'training_time': training_time,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"  ✗ {name} failed: {str(e)}")
        return {
            'name': name,
            'error': str(e),
            'status': 'failed'
        }

def train_all_models(models: Dict, X_train: np.ndarray, y_train: np.ndarray, use_parallel: bool = True) -> Dict:
    # Train all models with GPU support and parallel CPU training
    device = setup_gpu()
    results = {}
    
    # Separate GPU and CPU models
    gpu_models = []
    cpu_models = []
    
    for name, model in models.items():
        if name in ['ANN', 'MDN', 'CVAE']:
            gpu_models.append((name, model, X_train, y_train, device))
        else:
            cpu_models.append((name, model, X_train, y_train, torch.device('cpu')))
    
    # Train CPU models in parallel
    if cpu_models and use_parallel:
        print(f"\nTraining {len(cpu_models)} CPU models in parallel...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            cpu_results = list(executor.map(train_single_model, cpu_models))
            
            for result in cpu_results:
                if result['status'] == 'success':
                    results[result['name']] = {
                        'model': result['model'],
                        'training_time': result['training_time']
                    }
                else:
                    results[result['name']] = {'error': result['error']}
    else:
        for task in cpu_models:
            result = train_single_model(task)
            if result['status'] == 'success':
                results[result['name']] = {
                    'model': result['model'],
                    'training_time': result['training_time']
                }
            else:
                results[result['name']] = {'error': result['error']}

    if gpu_models:
        print(f"\nTraining {len(gpu_models)} GPU models...")
        for task in gpu_models:
            result = train_single_model(task)
            if result['status'] == 'success':
                results[result['name']] = {
                    'model': result['model'],
                    'training_time': result['training_time']
                }
            else:
                results[result['name']] = {'error': result['error']}
    
    # Show GPU memory if used
    if torch.cuda.is_available() and gpu_models:
        print(f"\nGPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
        torch.cuda.empty_cache()
    
    return results