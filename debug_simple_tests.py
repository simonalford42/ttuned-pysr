import numpy as np
import time
from basic_sr import BasicSR

# Define very simple test cases
def test_identity(seed):
    """y = x"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-5, 5, size=(20, 1))
    x = X[:, 0]
    y = x
    return X, y

def test_squared(seed):
    """y = x^2"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-5, 5, size=(20, 1))
    x = X[:, 0]
    y = x**2
    return X, y

def test_constant(seed):
    """y = 5"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-5, 5, size=(20, 1))
    y = np.ones(X.shape[0]) * 5
    return X, y

def test_linear(seed):
    """y = 2x + 3"""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-5, 5, size=(20, 1))
    x = X[:, 0]
    y = 2*x + 3
    return X, y

# Add print statement to the fitness function in BasicSR
class DebugBasicSR(BasicSR):
    def fitness(self, individual, X, y):
        """Calculate fitness as negative MSE with parsimony penalty"""
        y_pred = individual.evaluate(X)
        
        # Check if prediction contains NaN or inf
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return float('-inf')  # Invalid expression
            
        # Calculate MSE
        try:
            # Normalize predictions if they're too large
            if np.max(np.abs(y_pred)) > 1e6:
                scale = np.max(np.abs(y_pred))
                y_pred = y_pred / scale
                
            # Calculate mean squared error
            mse = np.mean((y - y_pred)**2)
            
            # If MSE is inf or NaN, return -inf
            if np.isinf(mse) or np.isnan(mse):
                return float('-inf')
                
            # Calculate complexity penalty
            complexity = individual.size()
            
            # Parsimony pressure (penalizing complexity)
            parsimony_penalty = self.parsimony_coefficient * complexity
            
            # Print debugging info for top expressions
            if mse < 1.0:  # Only print low MSE expressions
                print(f"Expr: {individual}, MSE: {mse:.6f}, Size: {complexity}, Fitness: {-mse - parsimony_penalty:.6f}")
            
            return -mse - parsimony_penalty
        except:
            return float('-inf')  # Invalid expression


def test_and_debug_model(test_name, test_func):
    print(f"\n=== Testing {test_name} ===")
    X, y = test_func(seed=42)
    
    # Print the actual data values
    print(f"Data sample (X, y):")
    for i in range(min(5, X.shape[0])):
        print(f"  {X[i]} -> {y[i]}")
    
    # Create model with debugging
    model = DebugBasicSR(
        population_size=100,
        tournament_size=3,
        num_generations=30,
        crossover_prob=0.7,
        mutation_prob=0.3,
        max_depth=3,  # Reduce depth for simpler expressions
        max_size=10,
        parsimony_coefficient=0.01  # Stronger parsimony pressure
    )
    
    # No normalization for debugging clarity
    start_time = time.time()
    model.fit(X, y)
    elapsed_time = time.time() - start_time
    
    # Check results
    if model.best_model_ is None:
        print(f"No valid model found for {test_name}")
        return
    
    # Evaluate performance
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred)**2)
    
    print(f"\nFinal results for {test_name}:")
    print(f"Best expression: {model.best_model_}")
    print(f"MSE: {mse:.6f}")
    print(f"Expression size: {model.best_model_.size()}")
    print(f"Time: {elapsed_time:.2f} seconds")
    
    # Print some sample predictions
    print("\nSample predictions vs actual:")
    for i in range(min(5, X.shape[0])):
        print(f"X: {X[i][0]:.2f}, Pred: {y_pred[i]:.2f}, Actual: {y[i]:.2f}, Error: {abs(y_pred[i]-y[i]):.2f}")


if __name__ == "__main__":
    # Run tests on very simple functions
    test_and_debug_model("y = x (identity)", test_identity)
    test_and_debug_model("y = x^2 (squared)", test_squared)
    test_and_debug_model("y = 5 (constant)", test_constant)
    test_and_debug_model("y = 2x + 3 (linear)", test_linear)