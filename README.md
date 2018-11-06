# BENPPy: BayesENproteomics in Python
Python implementation of BayesENproteomics.

BayesENproteomics fits user-specified regression models of arbitrary complexity to accurately model protein and post-translational modification fold changes in label-free proteomics experiments. BayesENproteomics uses Elastic Net regularization and observation weighting based on residual size and peptide identification confidence, implemented via MCMC sampling from conditional distributions, to prevent overfitting.

The initial proof-of-concept is described in our [preprint](https://www.biorxiv.org/content/early/2018/05/10/295527).

## Additonal features over BayesENproteomics Matlab implementation:
  * User-customised regression models to facilitate analysis of complex (or simple) experimental setups.
  * Protein and PTM run-level quantification (in addition to linear model fold change estimates) based on summation of user-specified effects.
  * No requirement to specify which PTMs to look for, BENPPy will automatically quantify any PTMs it can find (ideal for quantifying results obtained from unconstrained peptide search engines).
  * Option to utilise PyMC3-based NUTS sampler to fit a single customised model to an entire dataset (as opposed to the default option to fit protein-specific models), allowing the use of shared peptides (at the cost of very high RAM and CPU requirements).
  * MaxQuant compatibility.
  * Control group error propagation when calculating significance, if desired.
  * Option to use Bayes Factors instead of p-values, if desired.
  
## Required libraries
BENPPy is tested on Python 3.6 and requires [PyMC3](https://docs.pymc.io/).
 - Both BENPPy and PyMC3 also have the following dependencies:
   - NumPy
   - SciPy
   - Pandas
   - Matplotlib
   - [Theano](http://deeplearning.net/software/theano/)

## Usage
[instructions here]
