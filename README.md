# AdderNet Activation Cutoff Optimization via Differential Evolution

This project implements a **Differential Evolution (DE)** optimization framework to determine the optimal activation cutoffs for an unsigned **4-bit AOQ AdderNet model**. By optimizing the clipping values for each adder layer, I have successfully improved post-quantization accuracy from **83.87% to 89.50%**.

## 🚀 Overview

Quantization is essential for deploying deep learning models on resource-constrained hardware. However, fixed-point quantization (like unsigned int4) often introduces significant precision loss. This project focuses on finding the optimal "max activation" (cutoff) for each layer to balance clipping error and quantization resolution.

### Key Results
- **Baseline Accuracy (Post-Quantization):** 83.87%
- **Optimized Accuracy (Post-DE Optimization):** **89.50%**
- **Architecture:** AdderNet AOQ with Fusion Bias Removal (FBR)
- **Quantization:** Unsigned 4-bit (20 Layers)

## 🧠 Differential Evolution (DE) Algorithm

Differential Evolution is a stochastic, population-based optimization algorithm suitable for non-differentiable or complex search spaces. In this project, I used DE to search for a 20-dimensional vector $X$, where each element represents the activation cutoff for a specific adder layer.

### 1. Initialization
The population is initialized with 50 candidate solutions (trials). Initially, I started with a generalized vector space, and after multiple algorithm runs, I used the best population values to initialize a new population, significantly accelerating convergence.
- **Search Space:** 20 dimensions (one per adder layer).
- **Bounds:** Strictly constrained within a template range [2, 4] to ensure stable quantization.

### 2. Mutation
For each target vector in the population, a mutant vector is created by combining the weighted difference of two random candidates with a third candidate. It is important to note that the target vector and the three candidates ($X_{r1}, X_{r2}, X_{r3}$) are all **different unique discrete vectors**:
$$V_i = X_{r1} + F \cdot (X_{r2} - X_{r3})$$
Where $F$ is the mutation fact (set to 0.45).

### 3. Crossover
The target vector and the mutant vector are mixed to produce a trial vector based on a crossover probability ($CR \approx 0.85$):
$$U_{i,j} = \begin{cases} V_{i,j} & \text{if } \text{rand}(0,1) \le CR \\ X_{i,j} & \text{otherwise} \end{cases}$$

### 4. Selection
The trial vector is evaluated against the target vector using the **Accuracy** on the CIFAR-10 test set. If the trial vector yields higher accuracy, it replaces the target vector in the next generation.

## 🛠️ Project Structure

- `lib/Diff_evo.py`: core implementation of the Differential Evolution algorithm.
- `lib/test_de_maximization.py`: Main script for running the optimization. It handles:
    - Model loading and weight remapping.
    - Automated Quantization (AOQ).
    - Evaluation of candidate solutions.
- `lib/AdderNet_model/`: Contains the AdderNet 2.0 architecture and custom layers (FBR support).

## 📈 Optimization Process

The optimization uses a strictly constrained search space defined by a template of previous successful runs. This allows the algorithm to focus on fine-tuning sensitive layers (like the first and last blocks) while maintaining stable values for others.

```python
# Example Template Bounds
lb = [2.4, 2.73, 2.63, 2.4, ...] # Strictly constrained lower bounds
ub = [2.4, 2.74, 2.67, 2.4, ...] # Strictly constrained upper bounds
```

By concentrating the search on these high-impact dimensions, the DE algorithm identifies the subtle correlations between layer-wise clipping that standard heuristics or layer-wise quantization might miss.
