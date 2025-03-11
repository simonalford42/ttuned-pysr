import numpy as np
import random
import copy
import operator
from problems import PROBLEMS


class Node:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        
    def __str__(self):
        if self.value in ['+', '-', '*', '/', 'sin', 'cos', 'exp']:
            if self.value in ['sin', 'cos', 'exp']:
                return f"{self.value}({self.left})"
            else:
                return f"({self.left} {self.value} {self.right})"
        else:
            return str(self.value)
            
    def copy(self):
        if self.left is None and self.right is None:
            return Node(self.value)
        elif self.right is None:  # Unary operator
            return Node(self.value, self.left.copy())
        else:
            return Node(self.value, self.left.copy(), self.right.copy())
            
    def evaluate(self, X):
        try:
            if isinstance(self.value, (int, float)):
                return np.full(X.shape[0], self.value)
            elif isinstance(self.value, str) and self.value.startswith('x'):
                var_idx = int(self.value[1:])
                return X[:, var_idx]
            elif self.value == '+':
                return self.left.evaluate(X) + self.right.evaluate(X)
            elif self.value == '-':
                return self.left.evaluate(X) - self.right.evaluate(X)
            elif self.value == '*':
                return self.left.evaluate(X) * self.right.evaluate(X)
            elif self.value == '/':
                denominator = self.right.evaluate(X)
                # Prevent division by zero
                mask = np.abs(denominator) < 1e-10
                if np.any(mask):
                    denominator = denominator.copy()  # Create a copy to avoid modifying the original
                    denominator[mask] = 1e-10 * np.sign(denominator[mask])
                    denominator[denominator == 0] = 1e-10  # Handle exact zeros
                return self.left.evaluate(X) / denominator
            elif self.value == 'sin':
                value = self.left.evaluate(X)
                # Clip large values to prevent numerical issues
                value = np.clip(value, -100, 100)
                return np.sin(value)
            elif self.value == 'cos':
                value = self.left.evaluate(X)
                # Clip large values to prevent numerical issues
                value = np.clip(value, -100, 100)
                return np.cos(value)
            elif self.value == 'exp':
                value = self.left.evaluate(X)
                # Clip large values to prevent overflow
                value = np.clip(value, -50, 50)
                return np.exp(value)
            else:
                raise ValueError(f"Unknown node type: {self.value}")
        except Exception as e:
            # Return NaN array on error
            return np.full(X.shape[0], np.nan)
            
    def size(self):
        if self.left is None and self.right is None:
            return 1
        elif self.right is None:  # Unary operator
            return 1 + self.left.size()
        else:
            return 1 + self.left.size() + self.right.size()
            
    def depth(self):
        if self.left is None and self.right is None:
            return 0
        elif self.right is None:  # Unary operator
            return 1 + self.left.depth()
        else:
            return 1 + max(self.left.depth(), self.right.depth())
            
    def get_all_nodes(self):
        """Return a list of all nodes in the tree"""
        if self.left is None and self.right is None:
            return [self]
        elif self.right is None:  # Unary operator
            return [self] + self.left.get_all_nodes()
        else:
            return [self] + self.left.get_all_nodes() + self.right.get_all_nodes()


class BasicSR:
    def __init__(self, 
                 operators=['+', '-', '*', '/'],
                 unary_operators=['sin', 'cos', 'exp'],
                 constants=[-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
                 population_size=200,
                 tournament_size=5,
                 num_generations=50,
                 crossover_prob=0.7,
                 mutation_prob=0.3,
                 max_depth=5,
                 max_size=30,
                 parsimony_coefficient=0.01,  # Increased to discourage complexity
                 variable_prob=0.7,           # Probability of choosing variables over constants
                 variable_only_init=0.5):     # Fraction of population initialized with variables only
        self.operators = operators
        self.unary_operators = unary_operators
        self.constants = constants
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_depth = max_depth
        self.max_size = max_size
        self.parsimony_coefficient = parsimony_coefficient  # For penalizing complexity
        self.variable_prob = variable_prob
        self.variable_only_init = variable_only_init
        self.best_model_ = None
        
    def init_population(self, X):
        """Initialize a random population of expressions"""
        population = []
        num_vars = X.shape[1]
        
        # Calculate how many individuals should use variable-only trees
        variable_only_count = int(self.population_size * self.variable_only_init)
        
        # Add simple expressions for each variable
        for var_idx in range(num_vars):
            # Add the variable itself
            population.append(Node(f"x{var_idx}"))
            
            # Add variable with a constant for basic operations (multiplication, addition)
            for const in [1.0, 2.0, 3.0]:
                if len(population) < self.population_size:
                    # x * const
                    population.append(Node('*', Node(f"x{var_idx}"), Node(const)))
                    # x + const
                    population.append(Node('+', Node(f"x{var_idx}"), Node(const)))
            
            # Add simple binary operations between variables
            for var_idx2 in range(num_vars):
                for op in ['+', '*']:  # Focus on addition and multiplication
                    if len(population) < self.population_size:
                        population.append(Node(op, Node(f"x{var_idx}"), Node(f"x{var_idx2}")))
                        
            # Add simple squared variables (x^2 approximation)
            if len(population) < self.population_size:
                population.append(Node('*', Node(f"x{var_idx}"), Node(f"x{var_idx}")))
        
        # Add some constants
        for const in self.constants:
            if len(population) < self.population_size:
                population.append(Node(const))
        
        # Fill part of the rest with variable-only trees (to encourage algebraic expressions)
        var_only_count = 0
        while len(population) < self.population_size and var_only_count < variable_only_count:
            depth = random.randint(1, self.max_depth)
            tree = self._random_tree(depth, num_vars, variable_only=True)
            population.append(tree)
            var_only_count += 1
            
        # Fill the rest with random trees
        while len(population) < self.population_size:
            depth = random.randint(1, self.max_depth)
            tree = self._random_tree(depth, num_vars)
            population.append(tree)
            
        return population
    
    def _random_tree(self, max_depth, num_vars, depth=0, variable_only=False):
        """Generate a random expression tree"""
        # Terminal node
        if depth >= max_depth or (depth > 0 and random.random() < 0.3):
            if variable_only or random.random() < self.variable_prob:  # Variable
                var_idx = random.randint(0, num_vars-1)
                return Node(f"x{var_idx}")
            else:  # Constant
                const = random.choice(self.constants)
                return Node(const)
        
        # Function node
        # Reduce probability of generating unary operators 
        # (especially trigonometric functions)
        if random.random() < 0.2 and self.unary_operators:  # Unary operator
            op = random.choice(self.unary_operators)
            left = self._random_tree(max_depth, num_vars, depth+1, variable_only)
            return Node(op, left)
        else:  # Binary operator
            op = random.choice(self.operators)
            left = self._random_tree(max_depth, num_vars, depth+1, variable_only)
            right = self._random_tree(max_depth, num_vars, depth+1, variable_only)
            return Node(op, left, right)
            
    def simplify_expression(self, individual):
        """Apply basic algebraic simplifications to the expression"""
        # This is a simple example of expression simplification
        # In a more complete implementation, more rules would be added
        
        # Make a copy of the individual
        simplified = individual.copy()
        
        # Apply simplification rules recursively to all nodes
        self._simplify_node(simplified)
        
        return simplified
        
    def _simplify_node(self, node):
        """Recursively simplify a node in the expression tree"""
        if node is None:
            return
            
        # Recursively simplify children first
        if node.left is not None:
            self._simplify_node(node.left)
        if node.right is not None:
            self._simplify_node(node.right)
            
        # Apply simplification rules
        
        # Rule: x * 0 = 0
        if node.value == '*' and (
            (isinstance(node.left.value, (int, float)) and node.left.value == 0) or
            (isinstance(node.right.value, (int, float)) and node.right.value == 0)
        ):
            node.value = 0
            node.left = None
            node.right = None
            
        # Rule: x * 1 = x
        elif node.value == '*' and isinstance(node.right.value, (int, float)) and node.right.value == 1:
            node.value = node.left.value
            node.right = node.left.right
            node.left = node.left.left
            
        # Rule: x + 0 = x
        elif node.value == '+' and isinstance(node.right.value, (int, float)) and node.right.value == 0:
            node.value = node.left.value
            node.right = node.left.right
            node.left = node.left.left
            
        # Rule: x - 0 = x
        elif node.value == '-' and isinstance(node.right.value, (int, float)) and node.right.value == 0:
            node.value = node.left.value
            node.right = node.left.right
            node.left = node.left.left
    
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
            
            return -mse - parsimony_penalty
        except:
            return float('-inf')  # Invalid expression
    
    def tournament_selection(self, population, fitnesses):
        """Select an individual using tournament selection"""
        tournament = random.sample(range(len(population)), self.tournament_size)
        winner_idx = tournament[0]
        for idx in tournament:
            if fitnesses[idx] > fitnesses[winner_idx]:
                winner_idx = idx
        return population[winner_idx]
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        # If either parent is too small, just return a copy of parent1
        if parent1.size() < 2 or parent2.size() < 2:
            return parent1.copy()
        
        # Get all nodes from both parents
        nodes1 = parent1.get_all_nodes()
        nodes2 = parent2.get_all_nodes()
        
        # Select random crossover points
        crossover_point1 = random.choice(nodes1)
        crossover_point2 = random.choice(nodes2)
        
        # Create copies of parents
        new_parent1 = parent1.copy()
        new_parent2 = parent2.copy()
        
        # Find the corresponding nodes in the copies
        new_nodes1 = new_parent1.get_all_nodes()
        for i, node in enumerate(nodes1):
            if node is crossover_point1:
                new_crossover_point1 = new_nodes1[i]
                break
        
        # Swap the subtrees
        new_crossover_point1.value = crossover_point2.value
        new_crossover_point1.left = copy.deepcopy(crossover_point2.left)
        new_crossover_point1.right = copy.deepcopy(crossover_point2.right)
        
        # Check if the offspring is too large or too deep
        if new_parent1.size() > self.max_size or new_parent1.depth() > self.max_depth:
            return parent1.copy()
        
        return new_parent1
    
    def mutate(self, individual, num_vars):
        """Mutate an individual"""
        # If the tree is too small, grow it
        if individual.size() < 2:
            return self._random_tree(self.max_depth, num_vars)
        
        # Make a copy of the individual
        new_individual = individual.copy()
        
        # Get all nodes
        nodes = new_individual.get_all_nodes()
        
        # Select a random node to mutate
        mutation_point = random.choice(nodes)
        
        # Determine mutation type
        mutation_type = random.random()
        
        if mutation_type < 0.3:  # Replace with terminal
            if random.random() < 0.5:
                mutation_point.value = f"x{random.randint(0, num_vars-1)}"
            else:
                mutation_point.value = random.choice(self.constants)
            mutation_point.left = None
            mutation_point.right = None
        elif mutation_type < 0.6:  # Replace with random subtree
            # Make sure we have a valid depth range
            max_allowed_depth = max(1, self.max_depth - mutation_point.depth())
            depth = random.randint(1, max_allowed_depth)
            new_subtree = self._random_tree(depth, num_vars)
            mutation_point.value = new_subtree.value
            mutation_point.left = new_subtree.left
            mutation_point.right = new_subtree.right
        else:  # Change operator
            if mutation_point.left is not None:  # Not a terminal
                if mutation_point.right is None:  # Unary operator
                    mutation_point.value = random.choice(self.unary_operators)
                else:  # Binary operator
                    mutation_point.value = random.choice(self.operators)
        
        # Check if the mutated individual is too large or too deep
        if new_individual.size() > self.max_size or new_individual.depth() > self.max_depth:
            return individual.copy()
        
        return new_individual
    
    def fit(self, X, y):
        """Fit the model to the data"""
        num_vars = X.shape[1]
        population = self.init_population(X)
        
        best_individual = None
        best_fitness = float('-inf')
        generations_no_improvement = 0
        
        for generation in range(self.num_generations):
            # Occasionally simplify expressions to prevent bloat
            if generation % 5 == 0 and generation > 0:
                population = [self.simplify_expression(ind) for ind in population]
            
            # Evaluate fitness for each individual
            fitnesses = [self.fitness(ind, X, y) for ind in population]
            
            # Find the best individual
            current_best_idx = np.argmax(fitnesses)
            current_best = population[current_best_idx]
            current_best_fitness = fitnesses[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_individual = current_best.copy()
                best_fitness = current_best_fitness
                generations_no_improvement = 0
            else:
                generations_no_improvement += 1
            
            # Create a new population
            new_population = []
            
            # Elitism: keep the best individual
            new_population.append(current_best.copy())
            
            # Generate the rest of the population
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_prob and len(population) > 1:
                    # Crossover
                    parent1 = self.tournament_selection(population, fitnesses)
                    parent2 = self.tournament_selection(population, fitnesses)
                    offspring = self.crossover(parent1, parent2)
                else:
                    # Mutation
                    parent = self.tournament_selection(population, fitnesses)
                    offspring = self.mutate(parent, num_vars)
                
                new_population.append(offspring)
            
            population = new_population
            
            # Print progress
            if (generation + 1) % 10 == 0:
                mse = -best_fitness - self.parsimony_coefficient * best_individual.size()  # Adjust for the parsimony penalty
                print(f"Generation {generation + 1}/{self.num_generations}, Best MSE: {mse:.6f}, Size: {best_individual.size()}, Expression: {best_individual}")
            
            # Early stopping if no improvement for many generations
            if generations_no_improvement >= 15:
                print(f"Early stopping at generation {generation + 1} due to no improvement for 15 generations")
                break
        
        # Final simplification
        if best_individual is not None:
            best_individual = self.simplify_expression(best_individual)
        
        self.best_model_ = best_individual
        return self
    
    def predict(self, X):
        """Make predictions using the best model"""
        return self.best_model_.evaluate(X)


def test_on_problems():
    results = []
    
    for i, problem in enumerate(PROBLEMS):
        print(f"\nTesting on problem {i+1}: {problem.__name__}")
        
        # Generate data
        X, y = problem(seed=42)
        
        # Run BasicSR
        model = BasicSR(
            population_size=300,
            tournament_size=7,
            num_generations=100,
            crossover_prob=0.8,
            mutation_prob=0.2,
            max_depth=6,
            max_size=40,
            parsimony_coefficient=0.001
        )
        
        # Scale y values for better numerical stability
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std == 0:
            y_std = 1
        y_normalized = (y - y_mean) / y_std
        
        model.fit(X, y_normalized)
        
        # Check if a valid model was found
        if model.best_model_ is None:
            print(f"No valid model found for {problem.__name__}")
            results.append({
                'problem': problem.__name__,
                'mse': float('inf'),
                'expression': "No valid model found"
            })
            continue
        
        # Evaluate performance
        y_pred_normalized = model.predict(X)
        y_pred = y_pred_normalized * y_std + y_mean  # Convert back to original scale
        mse = np.mean((y - y_pred)**2)
        
        print(f"Final MSE: {mse:.6f}")
        print(f"Best expression: {model.best_model_}")
        print(f"Expression size: {model.best_model_.size()}")
        
        # Transform the expression to account for normalization
        expression_str = str(model.best_model_)
        adjusted_expression = f"({expression_str}) * {y_std} + {y_mean}"
        
        results.append({
            'problem': problem.__name__,
            'mse': mse,
            'expression': expression_str,
            'adjusted_expression': adjusted_expression,
            'size': model.best_model_.size()
        })
    
    # Print summary of results
    print("\n=== Summary of Results ===")
    for result in results:
        print(f"{result['problem']}: MSE = {result['mse']:.6f}, Size = {result['size']}")
        print(f"  Expression: {result['expression']}")
        print(f"  Adjusted expression: {result['adjusted_expression']}")


if __name__ == "__main__":
    test_on_problems()