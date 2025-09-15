import numpy as np
import random
import json
import os
import time
from datetime import datetime
from problems import ULTRA_SIMPLE_PROBLEMS, SIMPLE_PROBLEMS, ALL_SIMPLE_PROBLEMS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from training.format_utils import format_context, format_population_with_fitness, format_inference_input


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


class BasicSR:
    def __init__(self,
                 population_size=100,
                 num_generations=50,
                 max_depth=4,
                 max_size=15,
                 tournament_size=3,
                 collect_trajectory=False,
                 time_limit=None,
                 early_stop=True,
                 early_stop_threshold=3e-16,
                 min_generations=0):
        self.population_size = population_size
        self.num_generations = num_generations
        self.max_depth = max_depth
        self.max_size = max_size
        self.tournament_size = tournament_size
        self.collect_trajectory = collect_trajectory
        self.time_limit = time_limit
        self.operators = ['+', '-', '*', '/']
        self.constants = [1.0, 2.0]
        self.best_model_ = None
        self.trajectory = []
        self.generation_count = 0
        self.best_progression = []  # Track (generation, expression, mse) when best improves
        self.early_stop = early_stop
        self.early_stop_threshold = early_stop_threshold
        self.min_generations = min_generations

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

    def generate_child_via_evolution(self, population, fitnesses, num_vars, crossover_prob=0.7):
        """Generate a single child using basic evolution operators (crossover or mutation)"""
        if np.random.random() < crossover_prob:  # Crossover
            parent1 = self.tournament_selection(population, fitnesses)
            parent2 = self.tournament_selection(population, fitnesses)
            child = self.crossover(parent1, parent2)
        else:  # Mutation
            parent = self.tournament_selection(population, fitnesses)
            child = self.mutate(parent, num_vars)
        return child

    def generate_new_population(self, population, fitnesses, best_individual, num_vars, generation=0):
        """Generate new population using BasicSR evolution operators"""
        new_population = []

        # Elitism: keep best
        new_population.append(best_individual.copy())

        # Generate rest through evolution
        while len(new_population) < self.population_size:
            child = self.generate_child_via_evolution(population, fitnesses, num_vars)
            new_population.append(child)

        return new_population

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
        start_time = time.time()

        # Reset trajectory and progression for new run
        if self.collect_trajectory:
            self.trajectory = []
            self.generation_count = 0
        self.best_progression = []  # Always reset progression tracking

        for generation in range(self.num_generations):
            # Check time limit
            if self.time_limit is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= self.time_limit:
                    print(f"Time limit reached ({self.time_limit}s). Stopping evolution.")
                    break

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

                # Track progression of best solutions
                self.best_progression.append({
                    'generation': generation,
                    'expression': str(best_individual),
                    'mse': float(mse),
                    'fitness': float(best_fitness),
                    'size': best_individual.size()
                })

                print(f"Gen {generation}: MSE={mse:.6f}, Size={best_individual.size()}, Expr={best_individual}")

                # Check for near-zero MSE early stopping
                # Honor early-stopping only after a minimum number of generations
                if self.early_stop and (generation + 1) >= self.min_generations and mse <= self.early_stop_threshold:
                    print(f"MSE reached near-zero ({mse:.2e}). Stopping evolution.")
                    break

            # print(f"Gen {generation}: MSE={mse:.6f}, Size={best_individual.size()}, Expr={best_individual}")

            # Create new population
            population = self.generate_new_population(population, fitnesses, best_individual, num_vars, generation)
            if self.collect_trajectory:
                self.generation_count += 1

        self.best_model_ = best_individual
        return self

    def predict(self, X):
        """Make predictions with the best model"""
        if self.best_model_ is None:
            raise ValueError("Model not fitted yet")
        return self.best_model_.evaluate(X)

    def get_iterations_to_convergence(self, mse_threshold=1e-6):
        """Get the generation number when MSE first dropped below threshold"""
        for entry in self.best_progression:
            if entry['mse'] < mse_threshold:
                return entry['generation']
        return None  # Never converged

    def get_final_mse(self):
        """Get the final best MSE achieved"""
        if not self.best_progression:
            return float('inf')
        return self.best_progression[-1]['mse']

    def get_progression_data(self):
        """Get the full progression data"""
        return self.best_progression.copy()

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
        model = BasicSR(
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


class NeuralSR(BasicSR):
    """Neural version of BasicSR that uses a trained transformer for population generation"""

    def __init__(self, model_path, tokenizer_name="EleutherAI/gpt-neo-1.3B", **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve a valid model directory: accept either a leaf model dir (with config.json)
        # or a parent training output dir that contains a "final_model" subfolder.
        def _resolve_model_dir(path: str) -> str:
            if os.path.isfile(os.path.join(path, "config.json")):
                return path
            # Prefer final_model if present
            fm = os.path.join(path, "final_model")
            if os.path.isfile(os.path.join(fm, "config.json")):
                return fm
            # Otherwise, scan subdirs for a config.json (e.g., checkpoint-* or others)
            candidates = []
            try:
                for name in sorted(os.listdir(path)):
                    sub = os.path.join(path, name)
                    if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "config.json")):
                        # Favor presence of model weights
                        has_weights = (
                            os.path.isfile(os.path.join(sub, "pytorch_model.bin")) or
                            os.path.isfile(os.path.join(sub, "model.safetensors"))
                        )
                        candidates.append((0 if name == "final_model" else (1 if has_weights else 2), sub))
            except Exception:
                pass
            if candidates:
                candidates.sort()
                return candidates[0][1]
            # Nothing found; return the original for clearer downstream error
            return path

        resolved_path = _resolve_model_dir(model_path)
        if resolved_path != model_path:
            # Keep attribute for reference/debugging
            self.model_path = resolved_path

        # Load model and tokenizer
        # Prefer tokenizer from the trained checkpoint to preserve special tokens and IDs
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        except Exception:
            # Fallback: load base tokenizer if checkpoint tokenizer is unavailable
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            # Ensure required special tokens exist
            special_tokens = {
                "additional_special_tokens": [
                    "<CONTEXT>", "<POPULATION>", "<FITNESS>", "<TARGET>"
                ]
            }
            self.tokenizer.add_special_tokens(special_tokens)

        # Make sure pad token is defined for generation
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

        # Initialize expression parser
        from expression_parser import ExpressionParser
        self.expr_parser = ExpressionParser()

        # Track neural suggestion statistics
        self.neural_suggestions_total = 0
        self.neural_suggestions_well_formed = 0

    def format_context_and_population(self, population, fitnesses, num_vars, generation=0):
        """Format context and population for the neural model"""
        # Format context using shared utility
        variables = [f"x{i}" for i in range(num_vars)]
        context = format_context(generation, variables, self.operators, self.constants)

        # Format population using shared utility
        expressions = [str(expr) for expr in population]
        population_str = format_population_with_fitness(expressions, fitnesses)

        return context, population_str

    def extract_first_expression(self, text):
        """Extract the first complete expression from generated text"""
        text = text.strip()
        if not text:
            return text

        # Handle simple terminal case
        if not text.startswith('('):
            # For terminals like "x0" or "1.0", take until first space or <FITNESS>
            parts = text.split()
            if parts:
                first_part = parts[0]
                # Remove any trailing special tokens
                for token in ["<FITNESS>", "<CONTEXT>", "<POPULATION>", "<TARGET>"]:
                    if token in first_part:
                        first_part = first_part.split(token)[0]
                return first_part
            return text

        # For parenthesized expressions, need to find matching closing paren
        paren_count = 0
        for i, char in enumerate(text):
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count == 0:
                    # Found the complete expression
                    return text[:i+1]

        # If we didn't find matching parens, return the whole thing
        return text

    def parse_generated_expression(self, text):
        """Parse a generated expression string into a Node tree"""
        text = text.strip()
        return self.expr_parser.parse(text)

    def generate_new_population(self, population, fitnesses, best_individual, num_vars, generation=0):
        """Generate new population using neural model predictions with parallel sampling"""
        new_population = []

        # Elitism: keep best
        new_population.append(best_individual.copy())

        # Format input for neural model
        context, population_str = self.format_context_and_population(population, fitnesses, num_vars, generation)

        # Calculate how many new members we need to generate
        num_to_generate = self.population_size - 1  # -1 for elitism

        # Create input prompt using shared formatting
        input_text = format_inference_input(self.tokenizer.bos_token, context, population_str)
        # print(input_text)

        # Tokenize and generate all members in parallel
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                num_return_sequences=num_to_generate,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Process all generated outputs in batch
        all_predicted_targets = []  # Collect all targets first

        for i in range(num_to_generate):
            # Decode only up to the first stop token
            prompt_len = inputs["input_ids"].shape[1]
            full_gen_ids = outputs[i, prompt_len:]

            # Identify stop token ids (eos/pad and our control tokens)
            eos_id = self.tokenizer.eos_token_id
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
            stop_token_texts = ["<TARGET>", "<CONTEXT>", "<POPULATION>", "<FITNESS>"]
            stop_token_ids = []
            for tok in stop_token_texts:
                try:
                    tid = self.tokenizer.convert_tokens_to_ids(tok)
                    if tid is not None and tid != self.tokenizer.unk_token_id:
                        stop_token_ids.append(tid)
                except Exception:
                    pass
            # Always include eos/pad ids
            for tid in [eos_id, pad_id]:
                if tid is not None:
                    stop_token_ids.append(tid)

            # Find first occurrence of any stop token id
            cut_idx = None
            for j, tid in enumerate(full_gen_ids.tolist()):
                if tid in stop_token_ids:
                    cut_idx = j
                    break
            gen_ids = full_gen_ids[:cut_idx] if cut_idx is not None else full_gen_ids

            # Decode, skipping special tokens to avoid <|endoftext|> noise
            generated = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            # Treat the generated text itself as the target part
            target_part = generated

            # If model redundantly emitted a control token in text, keep only the part before it
            for tok in stop_token_texts:
                if tok in target_part:
                    target_part = target_part.split(tok, 1)[0].strip()

            # If still empty, fallback to evolution
            if not target_part:
                self.neural_suggestions_total += 1
                child = self.generate_child_via_evolution(population, fitnesses, num_vars)
                new_population.append(child)
                continue

            # Extract just the first complete expression
            first_expr = self.extract_first_expression(target_part)
            all_predicted_targets.append(first_expr if first_expr else generated)

        # Print all predicted targets on one line separated by spaces
        print("PREDICTED TARGETS:", " ".join(all_predicted_targets))

        # Now parse and add to population
        for i, first_expr in enumerate(all_predicted_targets):
            # Track neural suggestion statistics
            self.neural_suggestions_total += 1

            try:
                new_expr = self.parse_generated_expression(first_expr)
                # Successfully parsed - this is well-formed (no size constraints for NeuralSR)
                self.neural_suggestions_well_formed += 1
                new_population.append(new_expr)
            except Exception:
                # print('Not well formed, falling back to basic evolution')
                # print(first_expr or generated)
                # Fallback to basic evolution if parsing fails
                child = self.generate_child_via_evolution(population, fitnesses, num_vars)
                new_population.append(child)

        return new_population

    def get_well_formed_percentage(self):
        """Get the percentage of neural suggestions that were well-formed"""
        if self.neural_suggestions_total == 0:
            return 0.0
        return (self.neural_suggestions_well_formed / self.neural_suggestions_total) * 100.0

    def save_trajectory(self, filename):
        """Save the collected trajectory to a JSON file with neural-specific metadata"""
        if not self.collect_trajectory:
            raise ValueError("Trajectory collection was not enabled. Set collect_trajectory=True when initializing.")

        os.makedirs('data', exist_ok=True)

        trajectory_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_type': 'NeuralSR',
                'model_path': self.model_path,
                'population_size': self.population_size,
                'num_generations': self.num_generations,
                'max_depth': self.max_depth,
                'max_size': self.max_size,
                'tournament_size': self.tournament_size,
                'total_generations_recorded': len(self.trajectory),
                'neural_suggestions_total': self.neural_suggestions_total,
                'neural_suggestions_well_formed': self.neural_suggestions_well_formed,
                'well_formed_percentage': self.get_well_formed_percentage()
            },
            'trajectory': self.trajectory
        }

        filepath = os.path.join('data', filename)
        with open(filepath, 'w') as f:
            json.dump(trajectory_data, f, indent=2)

        print(f"Neural trajectory saved to {filepath}")
        print(f"Well-formed neural suggestions: {self.neural_suggestions_well_formed}/{self.neural_suggestions_total} ({self.get_well_formed_percentage():.1f}%)")
        return filepath


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
