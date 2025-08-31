import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
import torch
from sklearn.metrics import mean_squared_error
from typing import Dict

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

try:
    from Scripts.training import load_ik_data, train_all_models
    from Scripts.Models.Machine_Learning import ANNModel, KNNModel, ELMModel, RandomForestModel, SVMModel, GPRModel, MDNModel, CVAEModel
    from Scripts.Models.Traditional import analytical_ik, jacobian_ik, sdls_ik
except ImportError:
    try:
        from training import load_ik_data, train_all_models
        from Models.Machine_Learning import ANNModel, KNNModel, ELMModel, RandomForestModel, SVMModel, GPRModel, MDNModel, CVAEModel
        from Models.Traditional import analytical_ik, jacobian_ik, sdls_ik
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the project root directory")

class Traditional:
    def __init__(self, method_name, ik_function, timeout_per_sample=0.5):
        self.name = method_name
        self.ik_function = ik_function
        self.timeout_per_sample = timeout_per_sample
        self.is_trained = True
        self.training_time = 0.0  # Traditional methods don't need training
        self.output_dim = None
        self.timeout_count = 0
        self.failure_count = 0
    
    def fit(self, X_train, y_train):
        self.training_time = 0.0
        self.output_dim = y_train.shape[1]
        self.is_trained = True
        return self
    
    def predict(self, X_test):
        results = []
        self.timeout_count = 0
        self.failure_count = 0
        
        print(f"  Running {self.name} on {len(X_test)} samples (timeout: {self.timeout_per_sample}s per sample)...")
        
        for i, pose in enumerate(X_test):
            try:
                # Prepare pose (limit to 6D and pad if needed)
                pose_6d = pose[:6] if len(pose) >= 6 else np.pad(pose, (0, 6-len(pose)), 'constant')
                
                # Call traditional IK with timeout check
                joint_angles = self._solve_with_timeout(self.output_dim, pose_6d)
                
                # Ensure correct output dimension
                if len(joint_angles) != self.output_dim:
                    joint_angles = np.resize(joint_angles, self.output_dim)
                
                results.append(joint_angles)
                
            except TimeoutError:
                self.timeout_count += 1
                # Use fallback random solution for timeout cases
                fallback = np.random.uniform(0, 2*np.pi, self.output_dim)
                results.append(fallback)
                
            except Exception as e:
                # Other failures also get fallback
                self.failure_count += 1
                fallback = np.random.uniform(0, 2*np.pi, self.output_dim)
                results.append(fallback)
        
        total_failures = self.timeout_count + self.failure_count
        success_rate = (len(X_test) - total_failures) / len(X_test) * 100
        print(f"    {self.name}: {success_rate:.1f}% success rate ({self.timeout_count} timeouts, {self.failure_count} other failures)")
        
        return np.array(results)
    
    def _solve_with_timeout(self, dof, pose):
        start_time = time.time()
        
        try:
            # Call the actual IK function
            result = self.ik_function(dof, pose)
            
            # Check if we exceeded the time limit
            elapsed = time.time() - start_time
            if elapsed > self.timeout_per_sample:
                raise TimeoutError(f"Exceeded {self.timeout_per_sample}s limit")
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            if elapsed > self.timeout_per_sample:
                raise TimeoutError(f"Timeout after {elapsed:.3f}s")
            else:
                # Re-raise the original exception
                raise e

def evaluate_all_models(trained_models: Dict, X_test: np.ndarray, y_test: np.ndarray, force_cpu: bool = True) -> Dict:
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
            
            # Force CPU for inference if requested (for ML models)
            if force_cpu and name in ['ANN', 'MDN', 'CVAE']:
                original_device = None
                if hasattr(model, 'model') and hasattr(model.model, 'cpu'):
                    original_device = model.device if hasattr(model, 'device') else 'cuda'
                    model.model = model.model.cpu()
                elif hasattr(model, 'to'):
                    model = model.to('cpu')
                elif hasattr(model, 'device'):
                    original_device = model.device
                    model.device = torch.device('cpu')
            
            # Timed prediction
            start_time = time.time()
            y_pred = model.predict(X_test)
            inference_time = time.time() - start_time
            
            # Restore original device if changed
            if force_cpu and name in ['ANN', 'MDN', 'CVAE'] and original_device:
                if original_device == 'cuda' and torch.cuda.is_available():
                    if hasattr(model, 'model') and hasattr(model.model, 'cuda'):
                        model.model = model.model.cuda()
                    if hasattr(model, 'device'):
                        model.device = torch.device('cuda')
            
            # Calculate performance metrics
            joint_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            position_rmse = joint_rmse  # For compatibility with existing analysis
            
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

def results_dataframe(evaluation_results: Dict, dof: int = None, model_type_mapping: Dict = None) -> pd.DataFrame:
    data = []
    
    for name, results in evaluation_results.items():
        if 'error' not in results:
            row = {
                'model': name,
                'position_rmse': results['position_rmse'],
                'joint_rmse': results['joint_rmse'],
                'training_time': results['training_time'],
                'inference_time': results['inference_time'],
                'inference_time_per_sample': results['inference_time_per_sample'],
                'status': 'success'
            }
            
            # Add DOF if provided
            if dof is not None:
                row['dof'] = dof
                
            # Add model type if mapping provided
            if model_type_mapping and name in model_type_mapping:
                row['model_type'] = model_type_mapping[name]
            elif name.endswith('_IK'):
                row['model_type'] = 'Traditional'
            else:
                row['model_type'] = 'ML'
                
            # Add timeout information for traditional methods
            if hasattr(results['model'], 'timeout_count'):
                row['timeout_count'] = results['model'].timeout_count
                row['failure_count'] = results['model'].failure_count
                
            data.append(row)
    
    return pd.DataFrame(data)

def create_ml_models(input_dim=6, output_dim=3):
    models = {}
    
    # Core models (always available)
    models['ANN'] = ANNModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=[64, 32], 
        epochs=20
    )
    models['KNN'] = KNNModel(n_neighbors=5)
    models['ELM'] = ELMModel(input_dim=input_dim, output_dim=output_dim, hidden_dim=50)
    models['RandomForest'] = RandomForestModel(n_estimators=25)
    
    # Optional models (may not be available in all environments)
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

def create_traditional_models(output_dim=3, timeout_analytical=0.2, timeout_jacobian=0.5, timeout_sdls=0.8):
    traditional_models = {
        'Analytical_IK': Traditional('Analytical', analytical_ik, timeout_analytical),
        'Jacobian_IK': Traditional('Jacobian', jacobian_ik, timeout_jacobian),
        'SDLS_IK': Traditional('SDLS', sdls_ik, timeout_sdls)
    }
    
    return traditional_models

def single_test(dof, models, data_path, results_path, sample_limit=None):
    print(f"Testing DOF={dof}...")
    
    # Load data
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
    df = results_dataframe(evaluation_results, dof=dof)
    
    if not df.empty:
        results_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path / f'dof_{dof}_results.csv', index=False)
        print(f"  ✓ Results saved for DOF={dof}")
    
    return df

def multiple_test(dof_range, models, data_path, results_path, sample_limit=None):
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