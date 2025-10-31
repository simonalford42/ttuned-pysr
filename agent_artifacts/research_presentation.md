# Transformer-Tuned Symbolic Regression: Neural Algorithmic Enhancement

## Project Overview

This research project explores using transformer neural networks to enhance symbolic regression algorithms, specifically implementing "neural algorithmic tuning" to optimize algorithm performance beyond traditional hyperparameter optimization. The core concept is to train transformers on execution traces of symbolic regression algorithms and then improve them via reinforcement learning.

## Research Context & Motivation

### Problem Statement
Complex algorithms like PySR (Python Symbolic Regression) have numerous tunable parameters, but traditional hyperparameter optimization only adjusts existing parameters. This project proposes using transformers to conduct higher-level "algorithmic tuning" by learning and improving the algorithm's core decision-making processes.

### Inspiration from Prior Work
- **Stream of Search (SoS)**: Learning to Search in Language - demonstrated 36% improvement through transformer imitation and fine-tuning
- **Beyond A***: Better Planning with Transformers - achieved 30% shorter execution traces through search dynamics bootstrapping
- Both papers showed successful neural enhancement of algorithmic search processes

## Relation to Sandia, and prior discussions
### Double Pendulum
- Two ways to approach double pendulum problem: (1) treat it like a regular SR problem, and run symbolic regression to learn to model a double pendulum, given input data. (2) create a "DSL" of mechanical parts, and use a simulator which provides output to the system based off collected simulation statistics.
### Incorporating context into symbolic regression using neural networks
- Bigger picture: "smart guided SR". Neural networks can take in context that informs how search for expressions should be conducted. For example, pass in a graph of plotted symbolic regression data as context to the neural network which suggests expressions for the new generation.
- For now: keeping it in a regular SR loop. But you could imagine replacing an evolutionary algorithm with other forms of equation generation. For example, a neural network could directly suggest models based off simulation feedback.
### Debugging mechanical toy
- This is a symbolic regression problem. However, simply applying evolutionary search to discover the model would not work. We want a neural network that can intelligently search over potential models, given the feedback of a simulator, background information about the system, or other pieces of information.
- The challenging part is having a viable simulator that can be integrated into an AI system. To start, we plan to validate our approach on simpler problems that provide a similar task.

## Methodology

### Core Approach
1. **Algorithm Imitation**: Train transformers on execution traces of symbolic regression algorithms
2. **Performance Matching**: Ensure transformer can replicate original algorithm performance
3. **Neural Enhancement**: Optimize transformer using reinforcement learning on benchmark tasks

### Implementation Strategy
The project uses a simplified Python-based symbolic regression implementation ("BasicSR") as a proof-of-concept before scaling to full PySR:

**Neural Integration Points:**
- Population generation decisions
- Expression mutation/crossover choices
- Tournament selection strategies
- Fitness-based population evolution

## System Overview

- BasicSR (this repo): Minimal evolutionary SR in Python with explicit, readable operators and population dynamics. Used to generate trajectories and baselines.
- Trajectory Collection: Record full population state per generation (expressions + fitness, best/avg/diversity) to create rich training data.
- One‑Step Transformer Training: Convert trajectories into supervised examples of the form “context + current population → next expression”. Train GPT‑Neo variants on these.
- NeuralSR: Drop‑in replacement for BasicSR’s population update that samples new expressions from a trained transformer in parallel; falls back to evolutionary ops when suggestions are invalid.
- Benchmark Problems: Tiered suite from ultra‑simple to harder multi‑variate/rational forms for development, evaluation, and training data.

### Data Format & Training
Training data represents algorithm states as:
```
<CONTEXT>x0,x1,x2 | +,-,*,/ | 1.0,2.0
<POPULATION>(x0+1.0) <FITNESS>-119.33 (x1*x0)-1.0 <FITNESS>-108.50 ... (population with fitnesses)
<TARGET> [target expressions]
```

## Experiments & Results

### BasicSR Development & Validation
- **Performance Results (60s per problem)**:
  - pythagorean_3d: Solved in 0.3s (MSE ≈ 7.45e-31)
  - quadratic_formula_discriminant: Solved in 1.1s (MSE ≈ 3.21e-31)
  - polynomial_product: Solved in 0.0s (MSE ≈ 1.49e-31)
  - compound_fraction: Unsolved (MSE = 3.11e-02)
  - surface_area_sphere_approx: Partial solution (MSE = 4.65e-01)

### Trajectory Collection System
- **Comprehensive data collection** for 5 challenging problems
- **Rich trajectory data** with full population evolution history
- **Standardized format** for neural network training
- **Population diversity metrics** and fitness statistics per generation

### Neural SR Framework Implementation
- **NeuralSR class** that uses trained transformer models for population generation
- **Parallel processing** - generates entire population simultaneously rather than sequentially
- **Robust fallbacks** when neural generation fails, ensuring seamless comparison
- **Parameter alignment** between neural and basic SR for fair evaluation

### Training Infrastructure
- **One-step prediction framework** using transformer architecture
- **Multiple model configurations**: tiny (256 hidden, 4 layers, 16m params) and standard (1024 hidden, 16 layers, 250m params)
- **Training pipeline** with data conversion, model training, and evaluation components
- **Integration with wandb** for experiment tracking and monitoring

## Technical Architecture

### Core Components
1. **BasicSR Algorithm**: Simplified evolutionary symbolic regression with configurable parameters
2. **NeuralSR Enhancement**: Transformer-based population generation with evolutionary fallbacks
3. **Training Framework**: One-step prediction model training on algorithm execution traces
4. **Evaluation System**: Comparative performance analysis between neural and traditional methods

### Data Processing Pipeline
1. **Trajectory Collection**: Extract execution traces from BasicSR runs
2. **Format Conversion**: Transform trajectories into training-ready format
3. **Model Training**: Train transformers on one-step prediction tasks
4. **Neural Integration**: Use trained models to guide algorithm decisions

## Current Status

### Completed Milestones
- ✅ BasicSR algorithm implementation and validation
- ✅ Comprehensive trajectory collection system
- ✅ Neural SR framework with parallel population generation
- ✅ Training infrastructure and data processing pipeline
- ✅ Model training configurations (tiny and standard architectures)
- ✅ Integration testing and performance benchmarking

### Technical Achievements
- **Efficient parallel generation**: Reduced inference calls from N to 1 per generation
- **Robust fallback mechanisms**: Seamless operation when neural generation fails
- **Flexible architecture**: Support for different model sizes and configurations
- **Comprehensive evaluation**: Fair comparison framework between approaches

### Ongoing Work
- Model training on collected trajectory data
- Performance evaluation on benchmark problems
- Reinforcement learning enhancement of trained models
- Analysis of neural vs traditional algorithm performance

## Future Directions

### Immediate Next Steps
1. **Complete model training** on collected trajectory dataset
2. **Evaluate neural SR performance** against BasicSR baseline
3. **Implement reinforcement learning** optimization phase
4. **Scale to full PySR integration** based on proof-of-concept results

### Long-term Research Goals
- **Interpretability analysis**: Understanding what neural enhancements learn
- **Domain generalization**: Testing on diverse symbolic regression problems
- **Computational efficiency**: Optimizing neural components for practical deployment
- **Algorithm insights**: Extracting interpretable improvements for traditional algorithms

## Impact & Applications

This research demonstrates a novel approach to algorithm enhancement that could extend beyond symbolic regression to other optimization algorithms. The methodology of training neural networks on algorithm execution traces and then improving them via reinforcement learning represents a general framework for "neural algorithmic tuning" that could enhance various computational search and optimization processes.

The project specifically targets improving symbolic regression - a critical tool in scientific discovery for finding mathematical expressions that describe observed data relationships. Enhanced symbolic regression could accelerate scientific discovery across physics, biology, chemistry, and engineering domains.

## Project Resources

- **Codebase**: Complete implementation with BasicSR, NeuralSR, training framework
- **Datasets**: Trajectory collections from multiple challenging symbolic regression problems
- **Training Infrastructure**: Configurable model training with experiment tracking
- **Evaluation Framework**: Comprehensive performance comparison tools
