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
BENPPy is tested on Python 3.6 and requires [PyMC3](https://docs.pymc.io/). Both BENPPy and PyMC3 also have the following dependencies:
   - NumPy
   - SciPy
   - Pandas
   - Matplotlib
   - [Theano](http://deeplearning.net/software/theano/)

## Installation

Assuming a standard Python installation with pip and git, BENPPy can be installed via:

`pip install git+https://github.com/VenkMallikarjun/BENPPy` or `pip install BENPPy`

BENPPy can be imported by:

`import BENPPy as bp`

Depending on your installation, you may need to specify the environment vabiable 'MKL_THREADING_LAYER' to be 'GNU' in your IDE using `env MKL_THREADING_LAYER=GNU`.

## Usage

### 1. Create a new BayesENproteomics instance (`new_instance`) using: 

::

    new_instance = bp.BayesENproteomics(output_name,    # String specifying a folder name within your working directory where output files will be stored (folder will be created if it doesn't already exist).
                                        form            # Can be either 'progenesis' (default) or 'maxquant' to specify the peptide list input format.
                                        )

### 2. Start the analysis with:

::

    new_instance.doAnalysis(normalisation_peptides, # String specifying name of .csv file in format specified by 'form' containing peptides to be used for normalisation (can be the same as 'experimental_peptides').
                            experimental_peptides,  # String specifying name of .csv file in format specified by 'form' containing peptides to be used in quantitation.
                            organism,               # String specifying organism name. Can be 'human', 'mouse' or any UniProt proteome ID.
                            othermains_bysample,    # String specifying name of .csv file specifying additional main effects, with levels specified for each sample, to be included in model fitting. Defaults to ''.
                            othermains_bypeptide,   # String specifying name of .csv file specifying additional main effects, with levels specified for each peptide, to be included in model fitting. Defaults to ''.
                            otherinteractors,       # Dictionary specifying additional interacting parameters (E.g. {'Interactor1':'Interactor2','Interactor1':'Interactor3'}). Order of interactors does not matter. Defaults to {}.
                            regression_method,      # Can be either 'protein' (default) to fit separate models for each protein, or 'dataset' to fit a single model for entire dataset.
                            normalisation_method,   # Can be either 'median' (default) to normalise by median subtraction following log transformation, or 'none' to perform no normalisation.
                            pepmin,                 # Scalar specifying minimum number of peptides to fit a model for a protein. Proteins with fewer than pepmin peptides will be ignored. Defaults to 3.
                            ProteinGrouping,        # If ProteinGrouping is set to True, will treat all proteins with the same gene name as a single entity using all available peptides, otherwise each one will be calculated separately.
                            peptide_BHFDR,          # Scalar FDR cutoff employed to filter peptides before analysis. Defaults to 0.2.
                            nDB,                    # Scalar denoting number of databases used. Only modify this value if using earlier versions of Progenesis (<3.0). Defaults to 1.
                            incSubject,             # Bool denoting whether or not to include subject/run terms in model to be fit. Defaults to False.
                            subQuantadd,            # List of strings denoting which parameters to add to the 'Treatment' values to give subject-level quantification. Defaults to [''].
                            ContGroup,              # Bool denoting whether treatment variable specified in experimental_peptides headers is treated as a single continuous variable rather than multiple levels of a categorical variable. Defaults to False.
                            )

* If `form = 'progenesis'` than `experimental_peptides` is simply the peptide (ion)-level output from Progenesis QI, both `experimental_peptides` and `normalisation_peptides` must be formatted the same. If `form = 'maxquant'` than `experimental_peptides` is a list containing the MaxQuant modificationSpecificPeptides.txt first and any [PTM]Sites.txt (E.g. `['modificationSpecificPeptides.txt','Oxidation (M)Sites.txt','Acetylation (K)Sites.txt']`) and `normalisation_peptides` takes the format of modificationSpecificPeptides.txt.

#### 2.1 Customised models
By default BENPPy fits the following regularised model for each protein:

    `log_2(MS1 Intensity) = T + P + T*P + e`

Where `T` and `P` are treatment and peptide effects recpectively and `T*P` represents the interaction between them, with `e` representing the error term.

If `'dataset'` is specified for the `regression_method` arguement, the following model is fit for the entire dataset:

    `log_2(MS1 Intensity) = R + T + P T*R + T*P + e`
    
Where `R` is the protein effect and `T*R` is the interaction between treatment and protein.

These basic models can be extended by the user as desired by the `othermains_bysample` and `othermains_bypeptide` parameters. These are specified as strings containing the names of .csv files which contain columns of categorical identifiers with headers in the first row. Examples of how to specify othermains_bysample and othermains_bypeptide can be found in the examples folder (testsamplemains.csv and testpeptidemains.csv, respectively). Additional interaction effects can be specified by a dictionary in the `otherinteractors` parameter.

BENPPy will perform both protein-, PTM- (if PTMs are in your dataset) and pathway-level quantification, exporting the respective results as .csv files as each step finishes.

### 3. Inspect results:

After `doAnalysis` finishes there will be several new properties added to the instance created in step 1 (and exported as .csv files to the folder specified by `output_name`).

`new_instance`

### 4. Quality-control plots:
[Soon]

### 5. Contrasts and significance testing