# BENPPy: BayesENproteomics in Python
Python implementation of BayesENproteomics.

version 2.1.3

BayesENproteomics fits user-specified regression models of arbitrary complexity to accurately model protein and post-translational modification fold changes in label-free proteomics experiments. BayesENproteomics uses Elastic Net regularization and observation weighting based on residual size and peptide identification confidence, implemented via MCMC sampling from conditional distributions, to prevent overfitting.

The initial proof-of-concept is described in our [preprint](https://www.biorxiv.org/content/early/2018/05/10/295527) and in the [Matlab version](https://github.com/VenkMallikarjun/BayesENproteomics).

## Additonal features over BayesENproteomics Matlab implementation:
  * User-customised regression models to facilitate analysis of complex (or simple) experimental setups.
  * Protein and PTM run-level quantification (in addition to linear model fold change estimates) based on summation of user-specified effects.
  * No requirement to specify which PTMs to look for, BENPPy will automatically quantify any PTMs it can find (ideal for quantifying results obtained from unconstrained peptide search engines).
  * MaxQuant compatibility.
  * Control group error propagation when calculating significance (Welch's t-test), if desired.
  * Option to use Bayes Factors instead of p-values, if desired.
  * Option to run multiple MCMC in parallel for each protein - may improve numerical stability and reproducibility.
  * Specify fixed and random effects.
  
## Required libraries
BENPPy is tested on Python 3.6 and has the following dependencies:
   - NumPy
   - SciPy
   - Pandas
   - Matplotlib
   - Sci-Kit Learn

## Installation

Assuming a standard Python installation with pip and git, BENPPy can be installed via:

`pip install git+https://github.com/VenkMallikarjun/BENPPy`

BENPPy can be imported by:

`import BENPPy as bp`

## Usage

### 1. Create a new BayesENproteomics instance (`new_instance`) using: 

::

    new_instance = bp.BayesENproteomics(output_name,    # String specifying a folder name within your working directory where output files will be stored (folder will be created if it doesn't already exist).
                                        form            # Can be either 'progenesis' (default) or 'maxquant' to specify the peptide list input format.
                                        update_databases, # Boolean denoting whether to download new versions of UniProt and Reactome, defaults to True.
                                        )

### 2. Start the analysis with:

::

    new_instance.doAnalysis(normalisation_peptides, # String specifying name of .csv file in format specified by 'form' containing peptides to be used for normalisation (can be the same as 'experimental_peptides').
                            experimental_peptides,  # String specifying name of .csv file in format specified by 'form' containing peptides to be used in quantitation.
                            organism,               # String specifying organism name. Can be 'human', 'mouse' or any UniProt proteome ID.
                            othermains_bysample,    # String specifying name of .csv file specifying additional main effects, with levels specified for each sample, to be included in model fitting. Defaults to ''.
                            othermains_bypeptide,   # String specifying name of .csv file specifying additional main effects, with levels specified for each peptide, to be included in model fitting. Defaults to ''.
                            otherinteractors,       # Dictionary specifying additional interacting parameters (E.g. {'Interactor1':'Interactor2','Interactor1':'Interactor3'}). Order of interactors does not matter. Defaults to {}.
                            regression_method,      # Can be either 'protein' (default) to fit separate models for each protein, or 'dataset' to fit a single model for entire dataset (depreciated).
                            normalisation_method,   # Can be either 'median' (default) to normalise by median subtraction following log transformation, or 'none' to perform no normalisation (also assumes values are already logged).
                            pepmin,                 # Scalar specifying minimum number of peptides to fit a model for a protein. Proteins with fewer than pepmin peptides will be ignored. Defaults to 3.
                            ProteinGrouping,        # If ProteinGrouping is set to True, will treat all proteins with the same gene name as a single entity using all available peptides, otherwise each one will be calculated separately.
                            peptide_BHFDR,          # Scalar FDR cutoff employed to filter peptides before analysis. Defaults to 0.2.
                            nDB,                    # Scalar denoting number of databases used. Only modify this value if using earlier versions of Progenesis (<3.0). Defaults to 1.
                            incSubject,             # Bool denoting whether or not to include subject/run terms in model to be fit. Defaults to False.
                            subQuantadd,            # List of strings denoting which parameters to add to the 'Treatment' values to give subject-level quantification. Defaults to [''].
                            ContGroup,              # Bool denoting whether treatment variable specified in experimental_peptides headers is treated as a single continuous variable rather than multiple levels of a categorical variable. Defaults to False.
                            random_effects,         # List of strings denoting which effects will be sampled from a Gaussian with a mean of 0. E.g. ['Peptide','Donor']. Defaults to ['All'].
                            nChains,                # Integer denoting how many chains to run for each protein. Chains are run in parallel. Defaults to 3.
                            )


* If `form = 'progenesis'` than `experimental_peptides` is simply the peptide (ion)-level output from Progenesis QI, both `experimental_peptides` and `normalisation_peptides` must be formatted the same. Do not include spectral counts. 
* If `form = 'maxquant'` than `experimental_peptides` is a list containing the MaxQuant Peptides.txt first and any [PTM]Sites.txt (E.g. `['Peptides.txt','Oxidation (M)Sites.txt','Acetylation (K)Sites.txt']`) and `normalisation_peptides` takes the format of `'Peptides.txt'`.


*** Note that as of 13/06/2019, I have noticed that setting `nChains` to > 1 can cause the program to hang indefinitely when run in Spyder. The current work-around for this is to run it in an external terminal if `nChains` > 1 is required. 



#### 2.1 Customised models
By default BENPPy fits the following regularised model for each protein:

    log_2(MS1 Intensity) = T + P + T*P + e

Where `T` and `P` are treatment and peptide effects recpectively and `T*P` represents the interaction between them, with `e` representing the error term.

These basic models can be extended by the user as desired by the `othermains_bysample` and `othermains_bypeptide` parameters. These are specified as strings containing the names of .csv files which contain columns of categorical identifiers with headers in the first row. Examples of how to specify othermains_bysample and othermains_bypeptide can be found in the examples folder (testsamplemains.csv and testpeptidemains.csv, respectively). Additional interaction effects can be specified by a dictionary in the `otherinteractors` parameter.

BENPPy will perform both protein-, PTM- (if PTMs are in your dataset) and pathway-level quantification, exporting the respective results as .csv files as each step finishes.


### 3. Inspect results:

After `doAnalysis` finishes there will be several new properties added to the instance created in step 1 (and exported as .csv files to the folder specified by `output_name`).

#### Preliminary analysis properties - input data
* `new_instance.input_table` provides information about parameters used in analysis.
* `new_instance.peptides_used` lists peptides that were used in subsequent analysis.
* `new_instance.allValues` gives all values (observed and average imputed) for each peptide.
* `new_instance.missing_peptides_idx` Boolean array denoting where missing values are in `new_instance.allValues`.
* `new_instance.UniProt` UniProt database used at time of analysis.
* `new_instance.longtable` long-form vector table used in creation of design matrices.

#### Summary fold changes
* `new_instance.protein_summary_quant` protein-level log2 abundances.
* `new_instance.ptm_summary_quant` ptm-level log2 abundances.
* `new_instance.pathway_summary_quant` pathway-level log2 abundances.

#### Subject-level quantifications

If `subQuantadd` arguement is used when `doAnalysis` is called, or if `incSubject = True` protein and ptm subject/run-level quantification will be provided in `new_instance.protein_subject_quant` and `new_instance.ptm_subject_quant`, respectively.

### 4. Quality-control plots:

* `new_instance.boxplots()` will create boxplots of logged protein-, PTM-, peptidoform and pathway-level abundances. Extremely large values indicate potential overfitting. Tightening (decreasing) the peptide FDR threshold (`peptide_BHFDR` arguement in `doAnalysis`) or decreasing model complexity may improve overfitting.
* If contrasts have been made (see [step 5](https://github.com/VenkMallikarjun/BENPPy/blob/master/README.md#5-contrasts-and-significance-testing)), `new_instance.volcanoes(plot_type = 'protein',residue = 'any')` will create protein-level volcano plots (log2(fold changes) vs. -log10(BHFDR)). `plot_type` can also be `'ptm'` to show all PTMs or a string denoting a specfic PTM type (as written in input peptide lists), or `'pathway'`. If `plot_type = 'ptm'`, `residue` can equal any string of single-letter amino acids to plot only PTMs on those residues (E.g. `residue = 'NQR'`).
* Protein fold change VS PTM fold change plot can be made with:


::

    newinstance.proteinVSptm(ptm_type,          # String deonting PTM type to graph (as written in peptide list input files).
                             residue = 'any',   # String containing single-letter amino acid denoting which residues to plot.
                             exp = 1,           # Index for numerator in fold change calculation. Defaults to 1 (i.e. second column).
                             ctrl = 0           # Index for denominator in fold change calculation. Defaults to 0 (i.e. first column).
                             )


### 5. Contrasts and significance testing

Significance testing comparing all treatments to a single control group can be performed using `new_instance.doContrasts()` as follows:

::

    new_instance.doContrasts(Contrasted,      # String denoting which contrasts to test. Can be either 'protein', 'ptm'or 'pathway'. Defaults to 'protein'.
                             ctrl,            # Int denoting which column to use as the control group. Defaults to 0, meaning the first column.
                             propagateErrors, # Bool deonting whether to propagate control group errors into experimental group errors (using square root of sum of squares) for t-statistic calculation (Welch's t-test). Defualts to False.
                             UseBayesFactors  # Bool denoting whether to use Bayes Factors rather than p-values. Still needs testing. Defaults to False.
                             )
                             
This will add the `Contrasted` dataframe property to `new_instance` that can be inspected and manipulated by `new_instance.Contrasted`.


### 6. Load previous analysis

A previous BayesENproteomics instance (we'll stick with our example, `new_instance`) can be loaded using `new_instance.load()`, provided that `new_instance` is defined as in [step 1](https://github.com/VenkMallikarjun/BENPPy/blob/master/README.md#1-create-a-new-bayesenproteomics-instance-new_instance-using). `new_instance.load()` will look for the folder created during [step 1](https://github.com/VenkMallikarjun/BENPPy/blob/master/README.md#1-create-a-new-bayesenproteomics-instance-new_instance-using) (named using the `output_name` arguement). If this folder cannot be found, an error will be raised.
