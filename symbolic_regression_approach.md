# Neural-Enhanced Symbolic Regression: A Hybrid Approach

## Abstract

We present a hybrid approach to symbolic regression that combines traditional genetic programming with neural networks to enhance the search for mathematical expressions. Our method, termed NeuralSR, uses a transformer model to guide population generation in an evolutionary algorithm, improving both the quality and efficiency of expression discovery.

## 1. Problem Definition

The symbolic regression problem can be formally defined as follows:

Given:
- Input data matrix **X** ∈ ℝⁿˣᵈ where n is the number of samples and d is the dimensionality
- Target vector **y** ∈ ℝⁿ
- Set of operators Ω = {+, −, ×, ÷}
- Set of constants C = {1.0, 2.0}

**Objective**: Find an expression f(x₀, x₁, ..., xₐ₋₁) such that f(**X**) ≈ **y**, where f is composed of variables, constants from C, and operators from Ω.

The quality of a candidate expression is measured by the mean squared error:

**MSE(f) = (1/n) ∑ᵢ₌₁ⁿ (yᵢ − f(xᵢ))²**

## 2. BasicSR Algorithm

Our baseline approach, BasicSR, implements a traditional genetic programming algorithm with the following components:

### 2.1 Representation
Expressions are represented as binary trees where:
- Internal nodes contain operators from Ω
- Leaf nodes contain variables (x₀, x₁, ...) or constants from C

### 2.2 Population Initialization
The initial population P₀ of size μ is constructed as:
1. Simple terminals: individual variables and constants
2. Linear combinations: x₀ + c, x₀ × c for all variables and constants
3. Quadratic terms: x₀ × x₀ for all variables
4. Random trees of bounded depth (≤ 4) and size (≤ 15)

### 2.3 Evolution Operations
For each generation t → t+1:

**Selection**: Tournament selection with tournament size k = 3
**Crossover**: Subtree swapping between two parents with probability p_c = 0.7
**Mutation**: Random node replacement with probability p_m = 0.3
**Elitism**: Best individual is preserved across generations

### 2.4 Fitness Function
The fitness of individual i is defined as:
**F(i) = −MSE(i) − λ × size(i)**

where λ = 0.01 is a complexity penalty coefficient.

## 3. NeuralSR Algorithm

NeuralSR extends BasicSR by replacing the standard population generation mechanism with neural guidance while preserving the overall evolutionary framework.

### 3.1 Neural Architecture
We employ a GPT-Neo transformer model fine-tuned on evolutionary trajectories. The model learns to predict promising expressions given:
- Problem context (variables, operators, constants)
- Current population state (expressions and their fitness values)

### 3.2 Training Data Format
The neural model is trained on sequences formatted as:
```
<CONTEXT>Variables: x₀, x₁, ...; Operators: +, −, ×, ÷; Constants: 1.0, 2.0
<POPULATION>expr₁ <FITNESS> f₁ expr₂ <FITNESS> f₂ ...
<TARGET>new_expr
```

### 3.3 Neural Population Generation
At each generation, instead of using traditional crossover and mutation:

1. **Context Encoding**: Format current population and fitness values
2. **Neural Generation**: Sample k−1 new expressions from the trained model (preserving elitism)
3. **Parsing & Validation**: Convert generated strings to expression trees
4. **Fallback Mechanism**: If parsing fails, use BasicSR operators
