import numpy as np
import random
from problems import ULTRA_SIMPLE_PROBLEMS, SIMPLE_PROBLEMS

# Import the Node class and MinimalSR from the updated file
from minimal_sr import Node, MinimalSR


def test_initial_vs_evolved():
    """Compare best from initial population vs final evolved result"""

    print("=== Initial Population vs Evolved Comparison ===")

    all_results = []

    # Test on both problem sets
    problem_sets = [
        (ULTRA_SIMPLE_PROBLEMS, "Ultra-Simple"),
        (SIMPLE_PROBLEMS, "Regular Simple")
    ]

    for problem_set, set_name in problem_sets:
        print(f"\n=== {set_name} Problems ===")

        for i, problem in enumerate(problem_set):
            print(f"\nTesting problem {i+1}: {problem.__name__}")

            # Generate data
            X, y = problem(seed=42)
            print(f"Problem: {problem.__doc__}")

            # Create model and get initial population
            model = MinimalSR(
                population_size=50,
                num_generations=30,
                max_depth=3,
                max_size=10
            )

            num_vars = X.shape[1]
            initial_population = model.create_initial_population(num_vars)

            # Evaluate initial population
            initial_fitnesses = [model.fitness(ind, X, y) for ind in initial_population]
            best_initial_idx = np.argmax(initial_fitnesses)
            best_initial = initial_population[best_initial_idx]

            # Calculate initial MSE
            y_pred_initial = best_initial.evaluate(X)
            mse_initial = np.mean((y - y_pred_initial)**2)

            print(f"Best from initial population:")
            print(f"  MSE: {mse_initial:.6f}")
            print(f"  Size: {best_initial.size()}")
            print(f"  Expression: {best_initial}")

            # Now run full evolution (but quietly)
            original_fit = model.fit

            def quiet_fit(X, y):
                """Silent version of fit method"""
                num_vars = X.shape[1]
                population = model.create_initial_population(num_vars)

                best_fitness = -float('inf')
                best_individual = None

                for generation in range(model.num_generations):
                    # Evaluate fitness
                    fitnesses = [model.fitness(ind, X, y) for ind in population]

                    # Track best
                    current_best_idx = np.argmax(fitnesses)
                    current_best_fitness = fitnesses[current_best_idx]

                    if current_best_fitness > best_fitness:
                        best_fitness = current_best_fitness
                        best_individual = population[current_best_idx].copy()

                    # Create new population
                    new_population = []
                    new_population.append(best_individual.copy())

                    while len(new_population) < model.population_size:
                        if random.random() < 0.7:
                            parent1 = model.tournament_selection(population, fitnesses)
                            parent2 = model.tournament_selection(population, fitnesses)
                            child = model.crossover(parent1, parent2)
                        else:
                            parent = model.tournament_selection(population, fitnesses)
                            child = model.mutate(parent, num_vars)

                        new_population.append(child)

                    population = new_population

                model.best_model_ = best_individual
                return model

            quiet_fit(X, y)

            # Calculate evolved MSE
            y_pred_evolved = model.predict(X)
            mse_evolved = np.mean((y - y_pred_evolved)**2)

            print(f"Best after evolution:")
            print(f"  MSE: {mse_evolved:.6f}")
            print(f"  Size: {model.best_model_.size()}")
            print(f"  Expression: {model.best_model_}")

            # Calculate improvement
            if mse_initial > 0:
                improvement = (mse_initial - mse_evolved) / mse_initial * 100
            else:
                improvement = 0 if mse_evolved == 0 else -float('inf')

            print(f"Improvement: {improvement:.1f}% MSE reduction")

            # Determine if evolution helped
            if mse_evolved < mse_initial * 0.95:  # At least 5% improvement
                evolution_status = "✓ Helped"
            elif mse_evolved < mse_initial * 1.05:  # Within 5%
                evolution_status = "≈ Same"
            else:
                evolution_status = "❌ Worse"

            print(f"Evolution status: {evolution_status}")
            print("-" * 60)

            all_results.append({
                'problem_set': set_name,
                'problem': problem.__name__,
                'mse_initial': mse_initial,
                'mse_evolved': mse_evolved,
                'expr_initial': str(best_initial),
                'expr_evolved': str(model.best_model_),
                'size_initial': best_initial.size(),
                'size_evolved': model.best_model_.size(),
                'improvement_pct': improvement,
                'evolution_status': evolution_status
            })

    return all_results


def analyze_results(results):
    """Analyze the comparison results"""

    print("\n" + "="*80)
    print("ANALYSIS: Initial Population vs Evolution")
    print("="*80)

    # Count evolution outcomes
    helped = sum(1 for r in results if "Helped" in r['evolution_status'])
    same = sum(1 for r in results if "Same" in r['evolution_status'])
    worse = sum(1 for r in results if "Worse" in r['evolution_status'])

    total = len(results)

    print(f"\nOverall Evolution Performance:")
    print(f"  Helped: {helped}/{total} ({helped/total*100:.1f}%)")
    print(f"  Same:   {same}/{total} ({same/total*100:.1f}%)")
    print(f"  Worse:  {worse}/{total} ({worse/total*100:.1f}%)")

    # Show biggest improvements
    improvements = [r for r in results if r['improvement_pct'] > 5]
    if improvements:
        improvements.sort(key=lambda x: x['improvement_pct'], reverse=True)
        print(f"\nBiggest Improvements:")
        for r in improvements[:5]:
            print(f"  {r['problem']}: {r['improvement_pct']:.1f}% (MSE: {r['mse_initial']:.3e} → {r['mse_evolved']:.3e})")

    # Show problems where evolution made things worse
    worse_results = [r for r in results if "Worse" in r['evolution_status']]
    if worse_results:
        print(f"\nProblems where evolution made things worse:")
        for r in worse_results:
            print(f"  {r['problem']}: {r['improvement_pct']:.1f}% (MSE: {r['mse_initial']:.3e} → {r['mse_evolved']:.3e})")

    # Perfect solutions analysis
    perfect_initial = sum(1 for r in results if r['mse_initial'] < 1e-10)
    perfect_evolved = sum(1 for r in results if r['mse_evolved'] < 1e-10)

    print(f"\nPerfect Solutions (MSE < 1e-10):")
    print(f"  Initial population: {perfect_initial}/{total}")
    print(f"  After evolution:    {perfect_evolved}/{total}")

    return {
        'total': total,
        'helped': helped,
        'same': same,
        'worse': worse,
        'perfect_initial': perfect_initial,
        'perfect_evolved': perfect_evolved,
        'improvements': improvements,
        'worse_results': worse_results
    }


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    results = test_initial_vs_evolved()
    analysis = analyze_results(results)

    # Print detailed table
    print("\n" + "="*120)
    print("DETAILED COMPARISON TABLE")
    print("="*120)
    print(f"{'Problem':<20} {'Initial MSE':<12} {'Evolved MSE':<12} {'Improvement':<12} {'Status':<10} {'Initial Expr':<25} {'Evolved Expr':<25}")
    print("-"*120)

    for r in results:
        print(f"{r['problem']:<20} {r['mse_initial']:<12.2e} {r['mse_evolved']:<12.2e} {r['improvement_pct']:<12.1f}% {r['evolution_status']:<10} {r['expr_initial'][:24]:<25} {r['expr_evolved'][:24]:<25}")
