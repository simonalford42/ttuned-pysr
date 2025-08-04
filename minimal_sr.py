import numpy as np
import random
import json
import os
from datetime import datetime
from simple_problems import ULTRA_SIMPLE_PROBLEMS, SIMPLE_PROBLEMS, ALL_SIMPLE_PROBLEMS


class Node:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def __str__(self):
        if self.left is None and self.right is None:
            return str(self.value)
        else:
            return f"({self.left} {self.value} {self.right})"

    def copy(self):
        if self.left is None and self.right is None:
            return Node(self.value)
        else:
            return Node(self.value, self.left.copy(), self.right.copy())

    def evaluate(self, X):
        """Evaluate the expression on input data X"""
        try:
            if isinstance(self.value, (int, float)):
                return np.full(X.shape[0], self.value)
            elif isinstance(self.value, str) and self.value.startswith('x'):
                var_idx = int(self.value[1:])
                if var_idx >= X.shape[1]:
                    return np.full(X.shape[0], np.nan)  # Invalid variable
                return X[:, var_idx]
            elif self.value == '+':
                return self.left.evaluate(X) + self.right.evaluate(X)
            elif self.value == '-':
                return self.left.evaluate(X) - self.right.evaluate(X)
            elif self.value == '*':
                return self.left.evaluate(X) * self.right.evaluate(X)
            elif self.value == '/':
                denominator = self.right.evaluate(X)
                # Simple division by zero protection
                denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
                return self.left.evaluate(X) / denominator
            else:
                return np.full(X.shape[0], np.nan)
        except:
            return np.full(X.shape[0], np.nan)

    def size(self):
        """Count total nodes in the tree"""
        if self.left is None and self.right is None:
            return 1
        else:
            return 1 + self.left.size() + self.right.size()


class MinimalSR:
    def __init__(self,
                 population_size=100,
                 num_generations=50,
                 max_depth=4,
                 max_size=15,
                 tournament_size=3,
                 collect_trajectory=False):
        self.population_size = population_size
        self.num_generations = num_generations
        self.max_depth = max_depth
        self.max_size = max_size
        self.tournament_size = tournament_size
        self.collect_trajectory = collect_trajectory
        self.operators = ['+', '-', '*', '/']
        self.constants = [1.0, 2.0]
        self.best_model_ = None
        self.trajectory = []
        self.generation_count = 0

    def create_terminal(self, num_vars):
        """Create a terminal node (variable or constant)"""
        if random.random() < 0.7:  # Prefer variables
            return Node(f"x{random.randint(0, num_vars-1)}")
        else:
            return Node(random.choice(self.constants))

    def create_random_tree(self, max_depth, num_vars, depth=0):
        """Create a random expression tree"""
        # Force terminal at max depth or with some probability
        if depth >= max_depth or (depth > 0 and random.random() < 0.3):
            return self.create_terminal(num_vars)

        # Create binary operation
        op = random.choice(self.operators)
        left = self.create_random_tree(max_depth, num_vars, depth + 1)
        right = self.create_random_tree(max_depth, num_vars, depth + 1)
        return Node(op, left, right)

    def create_initial_population(self, num_vars):
        """Create initial population with diverse simple expressions"""
        population = []

        # Add very simple expressions first
        for i in range(num_vars):
            population.append(Node(f"x{i}"))  # Just x0, x1, etc.

        for const in self.constants:
            population.append(Node(const))  # Just constants

        # Add simple combinations
        for i in range(num_vars):
            for const in self.constants:
                population.append(Node('+', Node(f"x{i}"), Node(const)))  # x + c
                population.append(Node('*', Node(f"x{i}"), Node(const)))  # x * c

        # Add x^2 approximations (x * x)
        for i in range(num_vars):
            population.append(Node('*', Node(f"x{i}"), Node(f"x{i}")))

        # Fill rest with random trees
        while len(population) < self.population_size:
            tree = self.create_random_tree(self.max_depth, num_vars)
            if tree.size() <= self.max_size:
                population.append(tree)

        return population[:self.population_size]

    def fitness(self, individual, X, y):
        """Calculate fitness as negative MSE"""
        y_pred = individual.evaluate(X)

        # Check for invalid predictions
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return -1e10

        # Calculate MSE
        mse = np.mean((y - y_pred)**2)

        if np.isnan(mse) or np.isinf(mse):
            return -1e10

        # Simple complexity penalty
        complexity_penalty = 0.01 * individual.size()

        return -mse - complexity_penalty

    def tournament_selection(self, population, fitnesses):
        """Select individual via tournament selection"""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        best_idx = max(tournament_indices, key=lambda i: fitnesses[i])
        return population[best_idx]

    def mutate(self, individual, num_vars):
        """Simple mutation: replace a random node"""
        new_individual = individual.copy()

        # Find all nodes
        def get_all_nodes(node):
            if node.left is None and node.right is None:
                return [node]
            else:
                nodes = [node]
                nodes.extend(get_all_nodes(node.left))
                nodes.extend(get_all_nodes(node.right))
                return nodes

        nodes = get_all_nodes(new_individual)
        target_node = random.choice(nodes)

        # Replace with terminal or simple operation
        if random.random() < 0.5:
            # Replace with terminal
            replacement = self.create_terminal(num_vars)
            target_node.value = replacement.value
            target_node.left = None
            target_node.right = None
        else:
            # Replace with small subtree
            replacement = self.create_random_tree(2, num_vars)  # Small subtree
            target_node.value = replacement.value
            target_node.left = replacement.left
            target_node.right = replacement.right

        # Check size constraint
        if new_individual.size() > self.max_size:
            return individual

        return new_individual

    def crossover(self, parent1, parent2):
        """Simple crossover: swap random subtrees"""
        def get_all_nodes(node):
            if node.left is None and node.right is None:
                return [node]
            else:
                nodes = [node]
                nodes.extend(get_all_nodes(node.left))
                nodes.extend(get_all_nodes(node.right))
                return nodes

        child = parent1.copy()

        # Get crossover points
        child_nodes = get_all_nodes(child)
        parent2_nodes = get_all_nodes(parent2)

        if len(child_nodes) == 0 or len(parent2_nodes) == 0:
            return child

        target_node = random.choice(child_nodes)
        source_node = random.choice(parent2_nodes)

        # Perform crossover
        target_node.value = source_node.value
        target_node.left = source_node.left.copy() if source_node.left else None
        target_node.right = source_node.right.copy() if source_node.right else None

        # Check size constraint
        if child.size() > self.max_size:
            return parent1

        return child
    
    def record_population_state(self, population, fitnesses, generation):
        """Record the current population state (only if collect_trajectory=True)"""
        if not self.collect_trajectory:
            return
            
        # Convert population to string representations for storage
        expressions = [str(ind) for ind in population]
        
        state = {
            'generation': generation,
            'population_size': len(population),
            'expressions': expressions,
            'fitnesses': fitnesses.tolist() if isinstance(fitnesses, np.ndarray) else fitnesses,
            'best_fitness': max(fitnesses),
            'best_expression': expressions[np.argmax(fitnesses)],
            'avg_fitness': np.mean(fitnesses),
            'population_diversity': len(set(expressions))  # Number of unique expressions
        }
        
        self.trajectory.append(state)

    def fit(self, X, y):
        """Evolve expressions to fit the data"""
        num_vars = X.shape[1]
        population = self.create_initial_population(num_vars)

        best_fitness = -float('inf')
        best_individual = None
        
        # Reset trajectory for new run
        if self.collect_trajectory:
            self.trajectory = []
            self.generation_count = 0

        print(f"Starting evolution with {len(population)} individuals")

        for generation in range(self.num_generations):
            # Evaluate fitness
            fitnesses = [self.fitness(ind, X, y) for ind in population]
            fitnesses = np.array(fitnesses)
            
            # Record current state (if trajectory collection is enabled)
            self.record_population_state(population, fitnesses, generation)

            # Track best
            current_best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()

                # Calculate actual MSE for reporting
                y_pred = best_individual.evaluate(X)
                mse = np.mean((y - y_pred)**2)
                print(f"Gen {generation}: MSE={mse:.6f}, Size={best_individual.size()}, Expr={best_individual}")

            # Create new population
            new_population = []

            # Elitism: keep best
            new_population.append(best_individual.copy())

            # Generate rest through evolution
            while len(new_population) < self.population_size:
                if np.random.random() < 0.7:  # Crossover
                    parent1 = self.tournament_selection(population, fitnesses)
                    parent2 = self.tournament_selection(population, fitnesses)
                    child = self.crossover(parent1, parent2)
                else:  # Mutation
                    parent = self.tournament_selection(population, fitnesses)
                    child = self.mutate(parent, num_vars)

                new_population.append(child)

            population = new_population
            if self.collect_trajectory:
                self.generation_count += 1

        self.best_model_ = best_individual
        return self

    def predict(self, X):
        """Make predictions with the best model"""
        if self.best_model_ is None:
            raise ValueError("Model not fitted yet")
        return self.best_model_.evaluate(X)
    
    def save_trajectory(self, filename):
        """Save the collected trajectory to a JSON file"""
        if not self.collect_trajectory:
            raise ValueError("Trajectory collection was not enabled. Set collect_trajectory=True when initializing.")
            
        os.makedirs('data', exist_ok=True)
        
        trajectory_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'population_size': self.population_size,
                'num_generations': self.num_generations,
                'max_depth': self.max_depth,
                'max_size': self.max_size,
                'tournament_size': self.tournament_size,
                'total_generations_recorded': len(self.trajectory)
            },
            'trajectory': self.trajectory
        }
        
        filepath = os.path.join('data', filename)
        with open(filepath, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        print(f"Trajectory saved to {filepath}")
        return filepath


def test_on_problems(problem_set, title):
    """Test on a set of problems"""
    print(f"=== {title} ===")

    results = []
    for i, problem in enumerate(problem_set):
        print(f"\nTesting problem {i+1}: {problem.__name__}")

        # Generate data
        X, y = problem(seed=42)
        print(f"Problem: {problem.__doc__}")
        print(f"Data shape: X={X.shape}, y range=[{y.min():.3f}, {y.max():.3f}]")

        # Fit model
        model = MinimalSR(
            population_size=50,
            num_generations=30,
            max_depth=3,
            max_size=10
        )

        model.fit(X, y)

        # Evaluate
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred)**2)

        print(f"Final result: MSE={mse:.6f}, Expression={model.best_model_}")
        print("-" * 50)

        results.append({
            'problem': problem.__name__,
            'mse': mse,
            'expression': str(model.best_model_),
            'size': model.best_model_.size()
        })

    return results


def test_on_ultra_simple():
    """Test on ultra-simple problems first"""
    return test_on_problems(ULTRA_SIMPLE_PROBLEMS, "Testing on Ultra-Simple Problems")

def test_on_simple():
    """Test on regular simple problems"""
    return test_on_problems(SIMPLE_PROBLEMS, "Testing on Regular Simple Problems")

if __name__ == "__main__":
    # Test ultra-simple first
    ultra_results = test_on_ultra_simple()

    print("\n" + "="*60)
    print("ULTRA-SIMPLE RESULTS SUMMARY:")
    for result in ultra_results:
        status = "✓ Perfect" if result['mse'] < 1e-10 else "⚠ Poor"
        print(f"{result['problem']}: MSE={result['mse']:.2e}, Size={result['size']} {status}")

    # Test regular problems
    print("\n")
    simple_results = test_on_simple()

    print("\n" + "="*60)
    print("SIMPLE PROBLEMS RESULTS SUMMARY:")
    for result in simple_results:
        if result['mse'] < 0.01:
            status = "✓ Excellent"
        elif result['mse'] < 1.0:
            status = "⚠ Good"
        elif result['mse'] < 10.0:
            status = "⚠ Poor"
        else:
            status = "❌ Failed"
        print(f"{result['problem']}: MSE={result['mse']:.2e}, Size={result['size']} {status}")
