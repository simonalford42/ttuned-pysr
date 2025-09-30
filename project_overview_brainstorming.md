Transformer-tuned PySR

alternate names:
- Neural algorithmic tuning
- Optimizing Transformer-Distilled Algorithms for Symbolic Regression

Complex algorithms like PySR have a lot of tuneable parameters. Hyperparameter optimization packages help automate the process of tuning these parameters: take a suite of dataset of tasks, and tune the hyperparameters to optimize the algorithm for performance on these tasks. Here is a proposed way to use transformers to conduct even higher level "algorithmic tuning" by tuning not just hyperparameters, but the whole algorithm itself:

1. Train a transformer on execution traces of the algorithm on tasks from the dataset.
    - By the end of this step, the transformer should be able to "match" the performance of the original algorithm by executing the same algorithm itself.
2. Optimize the transformer using reinforcement learning on the benchmark.
    - After this step, the transformer should be even better at solving problems

Existing papers where this has been done:

Stream of Search (SoS): Learning to Search in Language: https://arxiv.org/pdf/2404.03683
- Countdown task: create target number using 4 input numbers and arithmetic ops. Example: inputs = 33, 66, 39, 13, target = 50. solution: 39 + 13 = 52. 66 / 33 = 2. 50 - 2 = 50.
- Generate dataset of "streams" of text of searching to solve the task using BFS/DFS/heuristics.
- Train a transformer to imitate the dataset
- Improve the transformer with RL, or with STaR: iteratively sample trajectories, filter to correct, and fine-tune on those.
- Fine-tuning improves performance by 36%

Beyond A∗: Better Planning with Transformers via
Search Dynamics Bootstrapping: https://arxiv.org/pdf/2402.14083
- Maze navigation and Sokoban with A*.
- Imitate traces of A*
- STaR: iteratively fine-tune on "shorter than before" discovered solutions.
- 30% shorter execution traces to find the solution.

Existing PySR algorithm:
- inputs: input variables, operators
- hyperparameters: # populations N , population size P, max expression size, max expression depth, tournament size S, selection prob p, crossover prob
- initialize N populations of size P
- with probability p, crossover two tournament winners. otherwise, mutate tournament winner.
- tournament
- choose S individuals and sort by fitness. Select individual according to geometric distribution: P(select i'th) = p * (1-p)^(i-1)
- copy & mutate selected individual, and if accepted with probability f(mutated fitness, original fitness) (simulated annealing), replace oldest member of population with this member
- repeat those steps for each population in parallel.
- wrap this all in an evolve-simplify-optimize loop.
- migrate between populations every so often

Goals of proposed transformer simplification:
- should be able to reproduce the original algorithm's performance
- should be able to explore alternative choices for things by increasing temperature of model
- should be able to optimize performance by training on "better" solutions

Plan:
- Represent each expression as a (expr, fitness) pair in context.
- Have LM decide whether to mutate or crossover
- Have LM decide which expression(s) to choose to mutate/crossover (tournament selection)
- Have LM decide how to mutate/crossover the expression(s)
- Everything else is the same.

Notes:
- LLM needs variables and operators at the beginning of context.

Simpler version:
1. start with python implementation of PySR that's simplified.
2. see if it works as a proof of concept (improves from the python version on performance on the benchmark)

Implementation:
1. Get benchmark
2. Run PySR on benchmark
3. Get "execution traces" of the things I'm concerned with for PySR
4. Figure out how to "insert" LM population modifications into PySR code (look at LaSR?)
5. Train LLM

Problems used for hyperparameter tuning for PySR:
https://github.com/MilesCranmer/pysr_wandb/blob/master/pysr_wandb/problems.py

Could be an easy "starting set" of problems to start from.

1. Run PySR on these problems
2. do a BasicSR basic SR python implementation, compare results to PySR
3. try to imitate the Python SR implementation, see how it compares

Basic Python SR algorithm implementation:
- Easy to implement
- Contains the core components of an evolutionary algorithm, without the implementation complexity of PySR
- Takes in operators, input/output data (same data format as PySR) and runs a simple evolutionary search to discover equations that best fit data
- Uses simplest mutations/tournaments/etc. to make the evolutionary search "work"

First step:
1. clone the repo of pysr problems
2. implement BasicSR algorithm in Python.
3. run BasicSR and PySR on the problems, compare accuracy and discovered equations
4. Setup code for extracting "algorithm traces" from BasicSR to train Transformer to imitate.
5. Setup code for training transformer on those traces. Take inspiration from processes used in streamofsearch and Beyond A* paper.
6. Train the transformer on the traces.
7. Write code for "executing" the transformer version of BasicSR, see how fast it runs, resulting performance.
8. "Fine-tune" the transformer using RL, expert iteration, etc. so that it improves the base SR algortihm using neurla primitives/hyperparameters.
9. See if it's improve version.
10. Iterate till that gets working. Use lessons learned to repeat the "train transformer on traces" for actual PySR.


# Thoughts 9/29/25

Maybe SR isn't the best application of this process? It'd be nice if there was another type of algorithm we could easily apply it to to show the applicability of the domain.
- something where the domain of the algorithm is easier to make a training set for
- it'd be nice if we could just directly use previous papers' training sets for a symbolic transformer, and then show that we do better by converting it into a search problem

Strategy:
1. find a prior work for algorithm that trains transformer to do it instead (like boolformer, etc)
2. show that if we take the same training set, and train transformer to do search, then it can do better
3. improve the search with RL

boolean expression finder (boolformer) could be another algorithmic application

It feels like this misses something of what we were aiming for with ttsr. maybe something kevin said?
- his main contribution is to be interested in using visual features. automated statistician.

this is a good benefit of our algorithm. by incorporating RL, we can train it to use context as guidance for the search.
originally the neural SR was going to be compared to code SR.
- try to solve more expressions faster.

it's much easier to compare to neural "direct SR" which will also take a while, and which we can do sampling, etc.

can I get boolformer and another SR paper running today?
- do some smaller training set of theirs that they used for experiments/ablations
- try doing the search based alternative for each one
- profit.

what about kevin's research, and making it similar to that?
- there's something about discrete search vs direct neural prediction.
- there are also connections to neural guided program synthesis
- think about wen ding's paper too.

my original goal was to be interested in making a "better than previous SR algorithm" thing. this could also be cool if we can get it to work. I think just using the same train/test datasets as the neural paper would be good though. I guess maybe I could make my own dataset generator too but ehhh.

What are the next steps?
1. do some literature review
    - what are the most important "direct SR transformer" papers? boolformer? any more recent stuff?
    - has anyone else looked into search with neural networks, especially as an alternative to direct prediction? both older stuff (like SAT solver net) and newer (inspired by o1/R1 reasoning transformers)
2. implement some experiments

https://arxiv.org/pdf/2309.12207
Boolformer

Stream of search.
Beyond A* paper.

https://storage.googleapis.com/deepmind-media/papers/SchultzAdamek24Mastering/SchultzAdamek24Mastering.pdf
This paper. "Mastering Board Games by External and Internal Planning with Language Models"

https://arxiv.org/abs/2304.03843
"Why think step by step? Reasoning emerges from the locality of experience"

https://proceedings.neurips.cc/paper_files/paper/2024/file/78f0db30c39c850de728c769f42fc903-Paper-Conference.pdf
