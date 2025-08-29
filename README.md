# vMC for t-J Model with BCS wave function on 3D Pyrochlore Lattice

## Description

This repository contains a variational quantum Monte Carlo (vQMC) code for simulating and analyzing a t-J model on a 3D pyrochlore lattice using variational Monte Carlo methods with stochastic reconfiguration optimization. The implementation uses a BCS-type wave function with complex variational parameters.

## Theory Background

### BCS Wave Function

The trial wave function for a system with ***L*** lattice sites and ***N*** up-spin and ***N*** down-spin electrons is based on the Bardeen-Cooper-Schrieffer (BCS) wave function:

![BCS](https://latex.codecogs.com/svg.image?\left|\psi\right\rangle=\exp\bigg(\sum_{ij}^L&space;f_{ij}\hat{c}^{\dagger}_{i,\uparrow}\hat{c}^{\dagger}_{j,\downarrow}\bigg)\left|0\right\rangle)

where ![c_up](https://latex.codecogs.com/svg.image?$$\hat{c}^{\dagger}_{i,\uparrow}$$) and ![c_do](https://latex.codecogs.com/svg.image?$$\hat{c}^{\dagger}_{i,\downarrow}$$) are creation operators for up-spin and down-spin electrons, respectively, and ![f_ij](https://latex.codecogs.com/svg.image?$$f_{ij}$$) is a matrix of variational parameters.

Each many-body configuration visited during the vMC sampling will be of the form:

![configuration](https://latex.codecogs.com/svg.image?|x\rangle=\hat{c}^{\dagger}_{R_1,\uparrow}\hat{c}^{\dagger}_{R_2,\uparrow}\dots\hat{c}^{\dagger}_{R_N,\uparrow}\hat{c}^{\dagger}_{S_1,\downarrow}\dots\hat{c}^{\dagger}_{S_N,\downarrow}|0\rangle)

where ![R_p](https://latex.codecogs.com/svg.image?R_p) and ![S_p](https://latex.codecogs.com/svg.image?S_p) represents the sites occupied by the up and down electrons respectively.

For each configuration of this kind, the amplitude of the wave function is given by:

![determinant](https://latex.codecogs.com/svg.image?$$\langle&space;x|\psi\rangle=\det&space;\big[f_{R_i,S_j}\big]$$)

where ![f_RiRj](https://latex.codecogs.com/svg.image?$$f_{R_iR_j}$$)  is the submatrix of   ![f_ij](https://latex.codecogs.com/svg.image?$$f_{ij}$$)  obtained by taking rows ![R_i](https://latex.codecogs.com/svg.image?R_i) and columns ![S_j](https://latex.codecogs.com/svg.image?S_j) .

### Fast Update Method

This implementation uses a fast-update scheme to efficiently compute wave function ratios between successive Monte Carlo configurations. Instead of recalculating the determinant (an ![OL3](https://latex.codecogs.com/svg.image?O(L^3)) operation), we compute and maintain the inverse of the ![f_RiRj](https://latex.codecogs.com/svg.image?$$f_{R_iR_j}$$) matrix, updating it incrementally as electron positions change during the Markov chain.

The update scheme depends on the move type:
- For hopping moves: rank-1 updates
- For spin-flip moves: rank-2 updates

For a deeper understanding of the fast-update scheme and its computational efficiency, please refer to Chapter 5, Section "5.6.2" (Fast Computation of the Determinants) of the book:

[**Quantum Monte Carlo Approaches for Correlated Systems** - Federico Becca, Sandro Sorella](https://www.cambridge.org/core/books/quantum-monte-carlo-approaches-for-correlated-systems/EB88C86BD9553A0738BDAE400D0B2900)

### Gutzwiller Projector

A Gutzwiller projector is implemented to eliminate double occupations on lattice sites, enforcing the strong correlation constraint of the model.

### Supported Models

This code can simulate two different models:
- **Heisenberg Model**: When there are no holes (all sites occupied)
- **t-J Model**: When holes are present in the system

## Code Structure

### C++ Simulation (`main_SR_lattice.cpp`)

Core quantum Monte Carlo implementation with the following features:
- MPI parallelization for distributed computing
- Stochastic Reconfiguration (SR) optimization of variational parameters
- Calculation of observables (energy, energy fluctuations, spin correlations)
 - Support for parameter optimization or fixed-parameter measurements

### Python Analysis Tools

    Statistical analysis scripts that:
    - Process simulation output data
    - Apply binning analysis to handle autocorrelations in Monte Carlo data
    - Calculate statistical averages and error estimates
    - Generate plots of observables

## Dependencies

### C++
    - Eigen (linear algebra library)
    - MPI (for parallelization)
    - C++11 compatible compiler

### Python
    - NumPy
    - Matplotlib
    - Numba

## Usage

### Running Simulations

    mpirun -np [NUMBER_OF_PROCESSES] ./main_SR_lattice [OUTPUT_DIRECTORY]

