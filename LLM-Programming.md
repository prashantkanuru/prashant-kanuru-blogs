# LLM Programming Language - Frameworking the ability to tap into latent-spaces of intelligence with descriptive language - aka parallel to SOFTWARE 3.0 but in Optimization Space
___

I intend to go through the various available frameworks which intend to program the LLM. To start with I have started looking at:
1. Optimization by Prompting (OPRO -Large Language Models as Optimizers)
2. Multi Prompt Instruction Proposal Optimizer (MIPRO - Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs)

## OPRO - Optimization by prompting
- Aim is to leverage **Large Language Models** as the optimizers, where the optimization task is described in natural language
- In each optimization step, the LLM generates new solutions from the prompt that contains previously generated solutions with their values, then the new solutions
are evaluated and added to the prompt

### Contextual Perspective of OPRO
- (Taken from the Paper) Optimization techniques are iterative, i.e. the optimization starts from an initial solution, then iteratively updates the solution to optimize the objective function.
- **For derivation-free optimization** optimization algorithm needs to be customized for an individual task to deal with the specific challenges posed by the decision space and the performance landscape.

### How does OPRO handle Optimization
- 
