# BENPPy: BayesENproteomics in Python
Python implementation of BayesENproteomics.

## Additonal features over BayesENproteomics Matlab implementation:
  1. User-customised regression models to facilitate analysis of complex (or simple) experimental setups.
  2. Protein and PTM run-level quantification (in addition to linear model fold change estimates) based on summation of user-specified effects.
  3. No requirement to specify which PTMs to look for, BENPPy will automatically quantify any PTMs it can find (ideal for quantifying results obtained from unconstrained peptide search engines).
  4. Option to utilise PyMC3-based NUTS sampler to fit a single customised model to an entire dataset (as opposed to protein-specific models currently implemented), allowing the use of shared peptides (at the cost of very high RAM and CPU requirements).
