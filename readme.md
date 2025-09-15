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


Further thoughts (7/31/25):
- How do we know that the transformer is still executing an algorithm?
- Should we do RL on the transformer executing the algorithm, or what about this alternative: implement the algorithm in code, and replace certain calls with neural heuristic calls, which have the same input/output but instead of using a simple implemented heuristic, use a learned neural heuristic. Can still train with reinforcement learning.
- This should be much more computationally efficient than fine-tuning the transformer it seems. idk.

Are there other algorithms it would be interesting to improve the heuristic for?
- Maybe we can learn heuristics that are good for certain data distributions, etc. For example, in maze finding, you could do better than L2 heuristic for A* if mazes have a certain structure. At the limit, you're doing a sort of description length thing. Lol see earlier thing with maze solving with A*... instead of searching according to heurstic, LLM just directly suggests new locations to try. result = much more sophisticated search than what's possible with following a heuristic. Don't limit the model!
- Instead of doing a new "proof of concept domain" I think the best is using it for PySR, because (1) we already know it'll kind of work, given prior work and principle, (2) pysr is the most interesting application. unless I can come up with another benchmark for the paper, I think just focus on PySR.

Plan:
- I think the "simple problems" were much better than the hard pysr hyperparam opt problems.
- We should do a proof of concept with BasicSR on a bench of easy problems. Maybe simple physics expressions, or something like that
- Should think about the extent to which the learned models could be "actually used" in PySR once we're done

Honestly, I like the following plan:
- Keep most of the PySR algorithm in place. Just train a neural heurstic for selection/mutation/etc.
- See how small we can make the NN while keep accuracy boost. Also, try to interpret the NN so that we can use the insights to improve the PySR algorithm.

Challenges:
- Improving the algorithm requires having a representative suite of data that our algorithm will be used on. However, this can be similar to actual PySR dev
- I don't know if I want to keep working with Miles or not.

But this seems like a cool project! And is related to my previous stuff. I should just make a good push to see if I can get a proof of concept working, and then go from there!
- Basic SR on toy problems. Do the whole pipeline quickly: collect data, train neural version, see if it can be improved, try to interpret it, etc.
- Goal for the end of today: get good basicSR data, set up training framework for neural version, try to get the accuracy to 100%.
- Maybe write a draft of what this paper would look like?
- Keep working on using claude code for this project.

Current status and plan for Basic SR:
- I think comparison_summary.md is newer than fixed_basicsr_comparison.md.
- Let's redo the analysis, where we run BasicSR and PySR on the set of simple problems only, getting rid of the other set of problems.
- before running basicsR and PySR, let's possibly modify the set of simple problems: (1) make sure they all can by synthesized with arithmetic ops + base variables. this means no constants besides 1 or 2 (assuming we provide 1 as a base variable), etc. (2) expand the set of problems to be 10 problems instead of 5. i think this means we'll only be searching for polynomials basically, is that right?
- Then, possibly simplify the BasicSR algorithm so that it only has to work for polynomials. Maybe there's some stuff we can get rid of in the implementation now that it's working iwth a more limited set of problems.


Hmm, it might be that making a reasonably good BasicSR algorithm is harder than getting it working on PySR. Maybe I should instead put some time into understanding feasibility and game plan for doing this on PySR.

Alright, made good progress! minimal_sr.py and improvement_trajectories.md and time_comparison_simple.py have the newest results. looks like our minimalSR algorithm is working quite nicely, and we can continue to improving it with an AI model! I feel very empowered with claude code, but I also spent $5 on it today.

Next steps:
1. commit changes to github
2. clean up the repo a bit.
1. set up pytorch lightning + wandb training framework. i'd like to collect


need to think about where we will insert LLM heuristics for the search. Plan from before still works:

Plan:
- Represent each expression as a (expr, fitness) pair in context.
- Have LM decide whether to mutate or crossover
- Have LM decide which expression(s) to choose to mutate/crossover (tournament selection)
- Have LM decide how to mutate/crossover the expression(s)
- Everything else is the same.

Possible things to control with LLM:
Possible levels:
1. given existing population and fitnesses, propose new population.
2. given existing population and fitnesses, propose a new individual. (Then query this len(population) times for new population)
    - more efficient: if we assume independent, then sample N individuals simultaneously.
3. given existing population and fitnesses, decide whether to crossover or mutate. Then propose parent(s), and the crossover/mutation of them.
4. given parents, output the crossover/mutation. (choice of crossover/mutation, and tournament selection, is done for you)
5. provide some structure to how crossover/mutation is done: given nodes, decide which ones to swap/mutate.

Principle: what approach minimizes number of LLM queries per iteration? I think approach 2 does. we can impose structure onto it (for example, each one predicts mutation/crossover/etc. simultaneously too.
- I wonder if when we do reinforcement learning, we can directly reward the proposals of that individual which worked out well too.


The higher the level of LLM control, the better performance is possible. But the harder optimization might be. OTOH, i could see the lowest level (level 5) being too granular to the point where there might not be much improvement possible.

What would the training data look like?
1. Sequences of population and fitnesses over time. LLM first learns to imitate the sequence, then we run it and see how the performance is (in distribution, OOD, on new tasks, etc.)
2. This one is basically the same as the above one if we do autoregressive completion. So let's combine those two.
3. This one can also be autoregressive where we add structure: "crossover? True\n" etc.
4. Again, I think all of this can be done with different prompt structures, and having different places where we query the model vs hard code different answers (need to make sure fine-tuning gradient descent works properly if we're inserting stuff during the completion though?)
The choice of how to mutate an individual like step 5 isn't interesting so not worth it if it's more complex to implement.

This is exciting! I want to work on this more.

Let's just start with the easiest one for now: sequences of population and fitness, and then generate the new population. But maybe make the code so that the other ones are possible too from the start.

To flesh this out a bit more:
- Make code that saves dataset of population+fitnesses trajectories from running BasicSR on some problem(s). Use it to save a dataset for each of our problems.
- Establish a training/validation/test set of SR problems. For now, these can be toy sets, once the method is working we'll switch to bigger sets. So for now just take the problems we have now and split them into train/test/split.
- To train an LLM on these sequences, I'm going to try to use the same code as stream of search repo (cloned  into stream-of-search).
- Look at how they structure their data, and then make some prompting to convert a trajectory dataset into LLM data that can be trained on.
- I'd like to start with much smaller models that train faster, so I can get a sense of if this can work. Modify so that the transformer is really small and trains fast. Try training on a small dataset and make sure train accuracy goes to 100%. Make sure we're logging all of the important metrics (hopefully stream of search already does the good stuff, but we might have to add more stuff)
- Stream of search did 5e5 training examples, 5e4 training steps. That's about 14-21 hours to run one training run.
- We'll have to set up how to use the model for eval (in a symbolic regression loop).
- We'll also have to set up doing the reinforcement learning to improve the model.
- It's clear that a big bottleneck for the experiments will be compute. Our ultimate goal is getting good performance on new symbolic regression tasks. Being able to handle more reinforcement learning would help with this. So smaller models, as much done in code instead of NN possible = better.

Can this project be applied to other projects?
- Improve adam optimizer
- Discover algorithms for other problems.

I'm a little obsessed with this recipe "train model, distill it into something interpretable, then you can use it to improve the algorithm". But there are other ways a system can be useful:
- it can learn from small amounts of data
- it can be a good model of human cognition

What to do now?
- Keep fleshing out details of the project, to see how long it'll take to train these models. Goal is to "see if it works" as quickly as possible.
- Sampling LLM completions is probably still doable with the trainer framework, but might be a bit more complicated. First chat with models to see if this is possible.
- Speed of sampling is really important though, so this seems really worth it to do.

Keep in mind that this is all just for BasicSR. We'll have to think about how to do all of this again for the "real" PySR algorithm, which will all be much harder.
- Maybe I should think about that a bit first too. Might impact how I implement all of this.

Key thing is just to look at the code for PySR and see how it does it, and where we could improve the search via a low-cost LLM query.

Another aspect that seems to be important is that I would prefer not to need a huge transformer. Think of it more like a neural patch.
Let's do this in parallel with "get something stupid working" to show as a proof of concept tomorrow:
- Trains on a train set, does well on validation set, learns faster because of learned heuristics.

Going back to what to do next ("get something working quickly" philosophy reinforced)
What is the smallest model we need to make this work?
- Generating syntactically correct expressions will be hard. We would actually need to train our model for a long time in order to get that.
- To help with this, lets stick with a small number of variables, and simple operators for now. Simple parsing so that almost anything can be parsed.
- overall, number of tokens will be small. just num variables + num operators and also number system for representing fitness (maybe special tokenization for this?)
- "train accuracy" cant be 100%, because of stochasticity in what the mutations/etc. are. ideally, we'd use a variational approach to model the randomness, but that's not the point.
- this is tricky though, not sure hwo to do this. I guess autoregressive kind of makes it work -- you eventualy anchor to one thing.

Whatever, let's just try a small model. We can evaluate models by sampling from them, and just measuring the error rate. We can still compare how long it takes to solve the task.
Can beef up the model size to decrease the error rate.

Next steps:
1. Make code that saves dataset of population+fitnesses trajectories from running BasicSR on some problem(s). Use it to save a dataset for each of our problems.
2. Establish a training/validation/test set of SR problems. For now, these can be toy sets, once the method is working we'll switch to bigger sets. So for now just take the problems we have now and split them into train/test/split.
3. To train an LLM on these sequences, I'm going to try to use the same code as stream of search repo (cloned  into stream-of-search).
4. Look at how they structure their data, and then make some prompting to convert a trajectory dataset into LLM data that can be trained on.
5. I'd like to start with much smaller models that train faster, so I can get a sense of if this can work. Modify so that the transformer is really small and trains fast.
6. We also want to modify so that instead of generating a whole new sequence, we train the LLM by sampling len(population) completions just given the initial prompt. Modify the construction of the training dataset and training procedure accordingly.
7. Try training on a small dataset and make sure train accuracy goes to 100%. Make sure we're logging all of the important metrics (hopefully stream of search already does the good stuff, but we might have to add more stuff)

Approach: to use stuff from stream of search repo, copy files into our repo.

alright, next lets try getting the stream of search training code working. based on readme.md info, can you make a first pass at copying over stream of search training files so that we can run the   │
│   same code on our trajectories collected from minimalSR implementation. the key things will be getting the training data format right. To start, just get the current stream of search working on our training data, and let us watch the

Integration stream of search new plan:
1. Copy over files, hook up our data. Make sure we can run training a really small model on our data, and view the results in wandb.
2. Change the training so that instead of full sequence completion, we sample len(population) from the first.

Need to add context to the LLM so that it knows what the constants, variables, and operators are when it is generating new expressions.
Here's how we'll do this, Format 1.0:

We'll add to convert_trajectories so there's ability to customize how the data is presented to the model. The easiest way

Question: what exactly is the model training on for stream of search? how does it differ for us? one advantage is that we don't need any context whatsoever beyond the current step.  For us, instead of having sequences, we really just have a big dataset of "predict one step" things:

given context, and current generation, predict the next generation.
context is O(1) (does not increase as we search for longer). To start, the context can just be ops, variables, and constants available. Later, we can experiment with including other things in the context: current iteration number, total iterations, amount of recent progress. We could even provide additional info for each expression, such as info on max/min value over range of input points, number of zeroes (like a text representation of what plotting would look like).

Hmm, given that we train models this way, maybe we can structure the training differently. First let's change convert_trajectories so that we save a dataset of one step predictions, and we can customize the context provided (for now, set up the context as specified)


- generalize to new problems
- give hints about problem in context: picture? mean/std, E[x_i * y], E[y^n]
  {
   "input": "x0,x1,x2 | +,-,*,/ | 1.0,2.0\n x0 <FITNESS>-119.33 x1(-108.50) x2(-112.03) 1.0(-91.63)
  2.0(-75.60) (x0+1.0)(-100.40) (x0*1.0)(-119.35) (x0+2.0)(-83.44) (x0*2.0)(-133.97) (x1+1.0)(-90.48)
  (x1*1.0)(-108.52) (x1+2.0)(-74.45) (x1*2.0)(-115.04) (x2+1.0)(-93.99) (x2*1.0)(-112.05) (x2+2.0)(-77.92)
  (x2*2.0)(-120.84) (x0*x0)(-63.39) (x1*x1)(-43.32) (x2*x2)(-54.51)\nNext:",
    "targets": [
      "(x1 * x1)",
      "((x2 / x1) - (1.0 + x0))",
      "(x0 + (x1 + 2.0))",
      "x2",
      "(2.0 * x0)",
      "((x0 + 2.0) * (x1 - x2))",
      "(x1 * x2)",
      "((2.0 / x1) * x0)",Gjk
      "(x0 + 1.0)",
      "(x2 * x1)",
      "1.0",
      "x2",
      "(x1 * x0)",
      "((1.0 * (1.0 / 2.0)) + 2.0)",
      "(x1 + 2.0)",
      "(x2 + 1.0)",
      "x0",
      "(x1 + x2)",
      "2.0",
      "(x1 * x1)"
    ],
    "metadata": {
      "problem": "pythagorean_3d",

      "generation": 0,
      "run_id": 1
    }
  }


Note a few things:
- right now i'm storing the data so that all the possible targets are saved in one point. Ultimately, we'd like to train the LLM with a "input/target" supervised learning, so we'll need to make another converter that duplicates len(target) points for each point in this version. Make this too, taking into account the best way to set up the training to do this type of training.

Note that the current training framework for stream of search probably does sequence prediction, but we'll probably have to change it to set up this slightly different style of training. Is that right?

1. make the convert trajectories
2. explore the current training framework, and how it differs from what we want, with me
3. make the dataset conversion to "training ready dataset" from the earlier conver_trajectories
4. change the training framework


Okay training framework is set up. Current pipeline is convert_trajectories -> train_one_step.py. Here are some changes to make for them/questions I have looking at the script:
- I didn't realize we will use special tokens for the setup to make the formatted input. Change convert_trajectories so it still makes a copy for each example, but just does json fields of 'context', 'population', and 'target'. should simplify the code in train_one_step tokenize function.
- Why aren't input_tokens variable used in tokenize?
- What is the attention mask?


# 8/11
Training a transformer model with the one-step training framework yields a model which can do the following:
- Given context and current population, predict a new expression

We can use this to generate a new population of size N by sampling N new expressions in parallel from the current population.
Then we can use this as the SR algorithm, and compare performance to the BasicSR algorithm.
Write code which loads a trained model (trained with train_one_step.py) and runs "neural SR" with it in this manner. In particular, you should
1. Refactor BasicSR so that the "given current population, propose new population" is abstracted into a function, for which there are two implementations: the BasicSR step, and the neural variant.
2. Make a way to run BasicSR with the neural injection. Compare neural/basic SR on the harder problems test bench, in terms of # of iterations to find a solution. also measure "wall clock" time for the two.


# 8/12
IN basic_sr.py, rewrite NeuralSR's generate_population so that it generates all members of the new population at the same time in parallel by sampling N outputs from the input string, and then processing them each.

# 8/26
Training the GPT neo tiny model is working, but doing it with the small model is not working.
Training the small model was done via train_one_step.py --config onestep-tiny.json
For the larger model, we had to switch to using acclerate for the distributed training. So, for example, in submit_jobs.sh, we'd submit:
`sbatch -J ttsr2 --partition ellis run2.sh accelerate launch --config_file training/accelerate.yaml training/train_one_step.py --config training/onestep-s.json`
I kept getting errors running this. For example, in out/49913.out, the error:

Traceback (most recent call last):
  File "/home/sca63/ttuned-pysr/training/train_one_step.py", line 266, in <module>
    main(args)
  File "/home/sca63/ttuned-pysr/training/train_one_step.py", line 254, in main
    trainer.train()
  File "/home/sca63/.conda/envs/ttsr/lib/python3.12/site-packages/transformers/trainer.py", line 2237, in train
    return inner_training_loop(
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/sca63/.conda/envs/ttsr/lib/python3.12/site-packages/transformers/trainer.py", line 2392, in _inner_training_loop
    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sca63/.conda/envs/ttsr/lib/python3.12/site-packages/accelerate/accelerator.py", line 1453, in prepare
    result = tuple(
             ^^^^^^
  File "/home/sca63/.conda/envs/ttsr/lib/python3.12/site-packages/accelerate/accelerator.py", line 1454, in <genexpr>
    self._prepare_one(obj, first_pass=True, device_placement=d) for obj, d in zip(args, device_placement)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sca63/.conda/envs/ttsr/lib/python3.12/site-packages/accelerate/accelerator.py", line 1302, in _prepare_one
    return self.prepare_model(obj, device_placement=device_placement)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sca63/.conda/envs/ttsr/lib/python3.12/site-packages/accelerate/accelerator.py", line 1669, in prepare_model
    model = torch.nn.parallel.DistributedDataParallel(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sca63/.conda/envs/ttsr/lib/python3.12/site-packages/torch/nn/parallel/distributed.py", line 835, in __init__
    _verify_param_shape_across_processes(self.process_group, parameters)
  File "/home/sca63/.conda/envs/ttsr/lib/python3.12/site-packages/torch/distributed/utils.py", line 282, in _verify_param_shape_across_processes
    return dist._verify_params_across_processes(process_group, tensors, logger)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: DDP expects same model across all ranks, but Rank 0 has 212 params, while rank 1 has inconsistent 0 params.

I tried debugging this with claude code and gpt codex, but didn't make any progress, kept getting the same error. Decided we should just directly read the code and see what's different between our distributed setup and the original distributed set up in the original stream-of-search repo. Looking at it, a lot is different! All of the config params are different.
Makes me think that I should just manually do the following:
1. run the stream of search train script, modifying it to only run for 10 minutes of gradient steps or so. check that it runs without issues on our cluster.
2. modify our script so that it's exactly the same set up as the stream of search script.

Look at the stream of search repo. Create a command to put in submit_jobs.sh that uses run2.sh to launch the job for the stream-of-search repo. Maybe it's easiest to copy submit_jobs.sh and run2.sh into the stream of search repo in the appropriate place, so that the paths work out more easily.

# Fixing BOS/EOS token issues for current trained model
It seems like there might be an issue with how my model is trained. I trained this model using training/train_one_step.py --config onestep-tiny.json. When I run neural_comparison.py, I get the following output (just a subset is shown):

--- Neural SR ---
Starting evolution with 20 individuals
Gen 0: MSE=43.292614, Size=3, Expr=(x1 * x1)
generated: (x0 * x1) <FITNESS>-43.32<TARGET>((x1 - x1) / (x0 - 2.0)) <FITNESS>-1014259.84<TARGET>(x2 + 2.0) <FITNESS>-
((x1 - x1) / (x0 - 2.0)) <FITNESS>-1014259.84
generated: x2 * 1.0) <FITNESS>-112.05<TARGET>x2 * (x0 * x0)) <FITNESS>-111.52<TARGET>(x1 * x1) <FITNESS>-43.32<TARGET>(x1 * x
x2 * (x0 * x0)) <FITNESS>-111.52
generated: x1 * x1) <FITNESS>-43.32 ((x0 + x0) + 1.0) <FITNESS>-100.40<TARGET>(x0 + 1.0) * 1.0) <FITNESS>-100.42
(x0 + 1.0) * 1.0) <FITNESS>-100.42
generated: (x1 * 2.0) <FITNESS>-115.04<TARGET>(x1 * x1) <FITNESS>-43.32<TARGET>x0 * x1) <FITNESS>-43.32<TARGET>(1.0 / x1)
(x1 * x1) <FITNESS>-43.32k


This seems wrong because I would expect from the training and training data that the model should only "complete" the input with a single target expression and then finish with an EOS token. But instead we see TARGET (expr) <FITNESS> (fitness) <TARGET> ... etc. which has issues of (1) target repeated multiple times, multiple expressions suggested, (2) fitness is included too.
Can you look at the trianing code and try to figure out why this is happening.


# 9/9 Evaluation
Training stuff is working a lot better now (i did some stuff not written about above). I'd like to work on evaluation. neural_comparison.py is a great comparison. I'd like to do a more thorough evaluation comparing Basic SR algorithm to the Neural SR algorithm. Please write an evaluation file called simple_evaluation.py which does the following:
- Does the basic vs neural comparison for all five "harder problems" from problems.py.
- For each problem, run N instances of Basic SR and N instances of Neural SR. Try to make Neural SR efficient (maybe we can put all N into one batch for each generation?). Then calcualte, print out and save to a json file statistics such as mean/std final MSE, avg # iterations to reach MSE ~0 (for those which succeeded), % that reached MSE ~0.
- Look at neural_comparison.py for inspiration for how to do this before doing it.
- If you have any other ideas for good metrics to include, run them by me before implementing.

# 9/9 overfitting training run.
Can you make a new dataset that is just a single trace from basic SR solving pythagorean 3d. I'd like to overfit to a tiny dataset to make sure things are working properly. Maybe read over this whole readme file to remember the files for where data is created.

# 9/12 generating a training set.
Up to now, the symbolic regression algorithm has been training and testing on the same problems. I'd now like to make a proper train/validation set, and then use the "harder problems" (from problems.py) as the test set.

The training data has two forms: first, the list of expressions that are targets. Second, the result of running BasicSR on these expressions. So, there are different parameters for the training data generation process, that we should store as metadata: the exact command run to generate the data, the max expression complexity, the list of operators/constants used, the number of expressions, seed, and the basicSR hyperparameters (number of expressions, seed, max number of generations, max depth, max size, others if needed). and others as appropriate.

The training set should be saved in two formats: the list of expressions, and the traces of BasicSR for those expressions.

Then we can split the set into training and validation sets. For the validation set, we can store both expressions and traces of expressions too.

Can you make an outline of files/function names, API, etc. for this, then if it looks good I'll have you implement it.

# 9/12 adding more context to SR algorithm
One of the advantages of training a neural network to propose expressions to be evaluated with SR is that the NN can incorporate rich context for guiding search. Typical SR "blindly" suggests new expressions by random mutations. Instead, we can try to have a smarter search. To do so, we can add additional information into the context (aka the input that the NN receives when it proposes new expressions).

See how context is created in format_utils.py. Now, let's add additional context for a problem. To start, we'll just add context for the target expression: visual/informational signals based off the input data. Can you modify format_utils.py so that there's the option of adding context when creating the dataset of examples. keep in mind how format_context function in format_utils is used by convert_trajectories.py. We should change convert_trjaectories too so it has an argument for the type of context. for now, do 'basic' context and 'rich' context.

for the rich context, let's add: (1) moments of the data, up to the third moment. (2) min and max of the data, (3) anything else we should add? please propose some ideas.

then we'll have 'superrich' context which also does a small plot of the data. add flags for this and a method stub which creates the small plot (in text) for the model, but don't implement it.
