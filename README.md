# Partial-Symbolic-Iterative-Neural-Reasoner
This repository contains the official implementation of the **Partial Symbolic Iterative Neural Reasoner (-NR)** pipeline, a metacognitive neurosymbolic architecture designed to solve complex Constraint Satisfaction Problems (CSPs) under incomplete rule conditions.

# Constraint Solving with Partial Symbolic Rules: A Neuro-Symbolic Pipeline (-NR)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

This repository contains the official implementation of the **Partial Symbolic Iterative Neural Reasoner (psi-NR)**, introduced in the paper *"Constraint Solving with Partial Symbolic Rules: A Neuro-Symbolic Pipeline."* Traditional Deep Learning (DL) architectures often struggle with complex Constraint Satisfaction Problems (CSPs). This repository demonstrates a novel, metacognitive neuro-symbolic pipeline that achieves near-perfect constraint satisfaction without requiring complete prior domain knowledge, evaluated on the **LinkedIn Queens** spatial puzzle.

##  Architecture Overview

The pipeline solves spatial constraints via an iterative loop consisting of three main components:
1. **Neuro-Solver:** A Transformer-based agent that outputs initial heuristic location predictions for the Queens.
2. **Mask-Predictor:** A neural network trained to identify deviations from expected structural layouts (rather than enforcing strict logical rules).
3. **Symbolic-Masker:** A modular verification layer that applies sub-selections of symbolic rules (Row, Column, Color, Adjacency) to strip away remaining violations before a final neural corrective pass.

##  Installation & Setup

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/edugoyu/partial-symbolic-iterative-neural-reasoner.git](https://github.com/edugoyu/partial-symbolic-iterative-neural-reasoner.git)
cd partial-symbolic-iterative-neural-reasoner
pip install -r requirements.txt
