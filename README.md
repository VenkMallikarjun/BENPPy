# BENPPy: BayesENproteomics in Python
Python implementation of BayesENproteomics.
BayesENproteomics fits user-specified regression models of arbitrary complexity to accurately model protein and post-translational modification fold changes in label-free proteomics experiments. BayesENproteomics uses Elastic Net regularization and observation weighting based on residual size and peptide identification confidence, implemented via MCMC sampling from conditional distributions, to prevent overfitting.

## Additonal features over BayesENproteomics Matlab implementation:
  1. User-customised regression models to facilitate analysis of complex (or simple) experimental setups.
  2. Protein and PTM run-level quantification (in addition to linear model fold change estimates) based on summation of user-specified effects.
  3. No requirement to specify which PTMs to look for, BENPPy will automatically quantify any PTMs it can find (ideal for quantifying results obtained from unconstrained peptide search engines).
  4. Option to utilise PyMC3-based NUTS sampler to fit a single customised model to an entire dataset (as opposed to protein-specific models currently implemented), allowing the use of shared peptides (at the cost of very high RAM and CPU requirements).
  5. MaxQuant compatibility.

## Usage
[instructions here]
