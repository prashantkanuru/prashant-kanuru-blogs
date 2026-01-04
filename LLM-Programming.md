---
layout: default
title: "Programming with LLMs: Software 3.0"
permalink: /llm-programming-language/
category: innovations
---

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

- Describe the optimization problem in natural language, then instructe the LLM to iteratively generate new solutions based on the problem description and the previously found solutions.
- Why LLMs - Generalization: Optimization with LLMs enables quick adaptation to different tasks by changing the problem description in the prompt, and the optimization process can be customized by adding instructions to specify the desired properties of the solutions.

### Optimization Problems tackled in the research paper

- Linear regression and travelling salesman problem  - two classical optimization problems that underpin many optimization problems in mathematics, computer science and operations research.
