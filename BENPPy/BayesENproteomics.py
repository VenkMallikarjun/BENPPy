import csv
import numpy as np
from scipy import stats
from scipy import linalg
from copy import deepcopy as dc
import urllib.request
import pandas as pd
import re
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.metrics import silhouette_samples, silhouette_score

'''
import pymc3 as pm
import theano.tensor as T
from theano.tensor.nlinalg import matrix_inverse
from theano import sparse
'''
import os
import multiprocessing
import time

__version__ = '2.6.3' #Fixed incorrect reporting of non-existant interaction effects in PTMsummaryQuant

# Output object that hold all results variables
class BayesENproteomics:

    # Create empty object to be filled anew with doAnalysis() or from a saved folder by load()
    def __init__(self, output_name = 'output', form = 'progenesis', update_databases = True):
        self.output_name = output_name
        self.form = form
        self.update_databases = update_databases
        if not os.path.exists(output_name):
            os.makedirs(output_name)

    # Wrapper for model fitting
    def doAnalysis(self, normalisation_peptides,
                   experimental_peptides,
                   organism,
                   othermains_bysample = '',
                   othermains_bypeptide = '',
                   otherinteractors = {},
                   regression_method = 'protein',
                   normalisation_method='median',
                   pepmin=3,
                   ProteinGrouping=False,
                   peptide_BHFDR=0.2,
                   nDB=1,
                   incSubject=False,
                   subQuantadd = [''],
                   ContGroup=[],
                   form='progenesis',
                   random_effects='all',
                   nChains=3,
                   impute='ami',
                   reassign_unreviewed=True,
                   continuousvars=[],
                   useReviewedOnly=False):

        print('continuousvars =',continuousvars,othermains_bysample)
        self.preprocessData(normalisation_peptides,
                            experimental_peptides,
                            organism,
                            pepmin,
                            continuousvars,
                            othermains_bysample,
                            othermains_bypeptide,
                            regression_method,
                            normalisation_method,
                            ProteinGrouping,
                            peptide_BHFDR,
                            nDB,
                            ContGroup,
                            impute,
                            reassign_unreviewed)

        self.doProteinAnalysis(otherinteractors,
                                pepmin,
                                incSubject,
                                subQuantadd,
                                self.form,
                                random_effects,
                                nChains,
                                continuousvars)

        self.doPathwayAnalysis(nChains, continuousvars)

    def preprocessData(self, normalisation_peptides,
                        experimental_peptides,
                        organism,
                        pepmin,
                        continuousvars,
                        othermains_bysample = '',
                        othermains_bypeptide = '',
                        regression_method = 'protein',
                        normalisation_method='median',
                        ProteinGrouping=False,
                        peptide_BHFDR=0.2,
                        nDB=1,
                        ContGroup=[],
                        impute='ami',
                        reassign_unreviewed=True,
                        useReviewedOnly=False):

        if regression_method == 'dataset':
            #otherinteractors['Protein_'] = 'Treatment'
            print('dataset method not supported in version 1.2.0 or higher, switching to protein method')
            regression_method = 'protein'
            #bayeslm = fitDatasetModel

        print('continuousvars =',continuousvars,othermains_bysample)
        peptides_used, longtable, UniProt, nGroups, nRuns = formatData(normalisation_peptides,
                                                                       experimental_peptides,
                                                                       organism,
                                                                       othermains_bysample,
                                                                       othermains_bypeptide,
                                                                       normalisation_method,
                                                                       ProteinGrouping,
                                                                       peptide_BHFDR,
                                                                       nDB,
                                                                       regression_method,
                                                                       ContGroup,
                                                                       self.form,
                                                                       self.update_databases,
                                                                       impute,
                                                                       reassign_unreviewed,
                                                                       useReviewedOnly)
        self.peptides_used = peptides_used
        #self.missing_peptides_idx = missing_peptides_idx
        self.UniProt = UniProt
        self.longtable = longtable

        if 'Group' in continuousvars:
            nGroups = 1

        print(nGroups,continuousvars)

        if self.form == 'maxquant' or self.form == 'peaks':
            input_table = pd.DataFrame({'group number':nGroups,
                                        'run number':nRuns,
                                        'minimum peptides':pepmin,
                                        'normalisation method':normalisation_method,
                                        'regression method':regression_method,
                                        'peptide FDR':peptide_BHFDR,
                                        'organism':organism,
                                        'experimental peptides':experimental_peptides,
                                        'normalisation peptides':normalisation_peptides})
        elif self.form == 'progenesis':
            input_table = pd.DataFrame({'group number':nGroups,
                                        'run number':nRuns,
                                        'minimum peptides':pepmin,
                                        'normalisation method':normalisation_method,
                                        'regression method':regression_method,
                                        'peptide FDR':peptide_BHFDR,
                                        'organism':organism,
                                        'experimental peptides':experimental_peptides,
                                        'normalisation peptides':normalisation_peptides},
                                        index = [0])

        self.input_table = input_table
        self.input_table.to_csv(self.output_name+'\\input_table.csv', encoding='utf-8', index=False,header=input_table.columns)
        self.peptides_used.to_csv(self.output_name+'\\peptides_used.tab', encoding='utf-8', sep="\t", index=False,na_rep='nan',header=peptides_used.columns)
        #self.input_table.to_csv(self.output_name+'\\input_table.csv', encoding='utf-8', index=False,header=input_table.columns)
        #self.missing_peptides_idx.to_csv(self.output_name+'\\missing_peptides_idx.csv', encoding='utf-8', index=False,header=missing_peptides_idx.columns)
        self.UniProt.to_csv(self.output_name+'\\UniProt.tab', encoding='utf-8', sep="\t", index=False,header=UniProt.columns)
        self.longtable.to_csv(self.output_name+'\\longtable.tab', encoding='utf-8', sep="\t", index=False,header=longtable.columns)


    def doProteinAnalysis(self, otherinteractors = {},
                            pepmin=3,
                            incSubject=False,
                            subQuantadd = [''],
                            form='progenesis',
                            random_effects='all',
                            nChains=3,
                            continuousvars=[]):

        bayeslm = fitProteinModels
        # Protein qunatification: fit protein or dataset model
        protein_summary_quant, PTM_summary_quant, isoform_summary_quant, protein_subject_quant, PTM_subject_quant, models, allValues, missingValues, OtherMains_table = bayeslm(self.longtable,
                                                                                                                                                                                otherinteractors,
                                                                                                                                                                                incSubject,
                                                                                                                                                                                subQuantadd,
                                                                                                                                                                                self.input_table['group number'][0],
                                                                                                                                                                                self.input_table['run number'][0],
                                                                                                                                                                                pepmin,
                                                                                                                                                                                random_effects,
                                                                                                                                                                                nChains,
                                                                                                                                                                                continuousvars)
        protein_list = list(protein_summary_quant.iloc[:,0].values.flatten())
        protein_info = pd.DataFrame(columns = self.UniProt.columns[[0,2,-2]])

        self.input_table['Variables used in subject quantification'] = pd.Series(subQuantadd)
        self.input_table.to_csv(self.output_name+'\\input_table.csv', encoding='utf-8', index=False,header=self.input_table.columns)
        self.missingValues = missingValues
        if self.form == 'progenesis':
            try:
                self.missingValues.columns = self.peptides_used.columns[13:]
            except:
                self.missingValues.columns = self.peptides_used.columns[17:]
        else:
            self.missingValues.columns = self.peptides_used.columns[11:]
        self.missingValues.to_csv(self.output_name+'\\missing_peptides_idx.csv', encoding='utf-8', index=False,header=missingValues.columns)

        # Append protein information used in pathway analysis
        for protein in protein_list:
            protein_info_idx = self.UniProt.iloc[:,1].isin([protein])
            if np.any(protein_info_idx):
                protein_ids = self.UniProt.loc[protein_info_idx,self.UniProt.columns[[0,2,-3]]]
                if protein_ids.shape[0] == 0:
                    protein_ids = pd.DataFrame([['NA','NA','NA']],columns = self.UniProt.columns[[0,2,-3]])
            else:
                protein_ids = pd.DataFrame([['NA','NA','NA']],columns = self.UniProt.columns[[0,2,-3]])

            protein_info = protein_info.append(protein_ids.iloc[0,:])

        #allValues.set_axis(['protein','peptide']+list(self.peptides_used.columns)[12:],axis=1,inplace=True)
        self.allValues = allValues
        #self.allValues.columns[2:] = self.peptides_used.columns[12:]
        self.protein_summary_quant = pd.concat((protein_summary_quant,protein_info.reset_index(drop=True)),axis=1,sort=False)
        self.protein_summary_quant = EBvar(self.protein_summary_quant)[0] # Empirical Bayes variance correction
        self.protein_summary_quant.to_csv(self.output_name+'\\protein_summary_quant.csv', encoding='utf-8', index=False,header=self.protein_summary_quant.columns)
        self.protein_subject_quant = pd.concat((protein_subject_quant,protein_info.reset_index(drop=True)),axis=1,sort=False)
        self.protein_subject_quant.to_csv(self.output_name+'\\protein_subject_quant.csv', encoding='utf-8', index=False,header=self.protein_subject_quant.columns)
        self.allValues.to_csv(self.output_name+'\\allValues.csv', encoding='utf-8', index=False,header=self.allValues.columns)
        self.other_summary_quant =  pd.concat((OtherMains_table,protein_info.reset_index(drop=True)),axis=1,sort=False)
        try:
            self.other_summary_quant = EBvar(self.other_summary_quant)[0]
        except:
            print('no other main effects.')
        self.other_summary_quant.to_csv(self.output_name+'\\other_summary_quant.csv', encoding='utf-8', index=False,header=self.other_summary_quant.columns)

        if PTM_summary_quant.shape[0] > 0:
            protein_list = list(PTM_summary_quant.iloc[:,1].values.flatten())
            protein_info = pd.DataFrame(columns = self.UniProt.columns[[0,2,-3]])
            for protein in protein_list:
                protein_info_idx = self.UniProt.iloc[:,1].isin([protein])
                if np.any(protein_info_idx):
                    protein_ids = self.UniProt.loc[protein_info_idx,self.UniProt.columns[[0,2,-3]]]
                    if protein_ids.shape[0] == 0:
                        protein_ids = pd.DataFrame([['NA','NA','NA']],columns = self.UniProt.columns[[0,2,-3]])
                else:
                    protein_ids = pd.DataFrame([['NA','NA','NA']],columns = self.UniProt.columns[[0,2,-3]])

                protein_info = protein_info.append(protein_ids.iloc[0,:])

            self.PTM_summary_quant = pd.concat((PTM_summary_quant,protein_info.reset_index(drop=True)),axis=1,sort=False)
            self.PTM_summary_quant = EBvar(self.PTM_summary_quant)[0] # Empirical Bayes variance correction
            self.PTM_summary_quant.to_csv(self.output_name+'\\PTM_summary_quant.csv', encoding='utf-8', index=False,header=self.PTM_summary_quant.columns)
            self.PTM_subject_quant = pd.concat((PTM_subject_quant,protein_info.reset_index(drop=True)),axis=1,sort=False)
            self.PTM_subject_quant.to_csv(self.output_name+'\\PTM_subject_quant.csv', encoding='utf-8', index=False,header=self.PTM_subject_quant.columns)

        if subQuantadd == [''] and incSubject == False:
            subQuant = pd.DataFrame()
        else:
            subQuant = self.protein_subject_quant

        protein_list = list(isoform_summary_quant.iloc[:,0].values.flatten())
        protein_info = pd.DataFrame(columns = self.UniProt.columns[[0,2,-3]])
        for protein in protein_list:
            protein_info_idx = self.UniProt.iloc[:,1].isin([protein])
            if np.any(protein_info_idx):
                protein_ids = self.UniProt.loc[protein_info_idx,self.UniProt.columns[[0,2,-3]]]
                if protein_ids.shape[0] == 0:
                    protein_ids = pd.DataFrame([['NA','NA','NA']],columns = self.UniProt.columns[[0,2,-3]])
            else:
                protein_ids = pd.DataFrame([['NA','NA','NA']],columns = self.UniProt.columns[[0,2,-3]])

            protein_info = protein_info.append(protein_ids.iloc[0,:])

        self.isoform_summary_quant = pd.concat((isoform_summary_quant,protein_info.reset_index(drop=True)),axis=1,sort=False)
        self.isoform_summary_quant = EBvar(self.isoform_summary_quant)[0] # Empirical Bayes variance correction
        self.isoform_summary_quant.to_csv(self.output_name+'\\isoform_summary_quant.csv', encoding='utf-8', index=False,header=self.isoform_summary_quant.columns)


    def doPathwayAnalysis(self, nChains = 3, continuousvars = []):
        # Pathway quantification: fit pathway models
        pathway_summary_quant, pathway_models, Reactome = fitPathwayModels(self.protein_summary_quant,
                                                                           self.UniProt,
                                                                           self.input_table['organism'][0],
                                                                           self.longtable,
                                                                           self.input_table['run number'][0],
                                                                           False,
                                                                           self.protein_subject_quant,
                                                                           self.update_databases,
                                                                           nChains,
                                                                           continuousvars)
        if pathway_summary_quant.shape[0] > 0:
            pathway_summary_quant = EBvar(pathway_summary_quant)[0] # Empirical Bayes variance correction
            self.pathway_summary_quant = pathway_summary_quant
            self.pathway_summary_quant.to_csv(self.output_name+'\\pathway_summary_quant.csv', encoding='utf-8', index=False,header=self.pathway_summary_quant.columns)

        self.Reactome = Reactome
        self.Reactome.to_csv(self.output_name+'\\Reactome.csv', encoding='utf-8', index=False,header=self.Reactome.columns)

    # Load exproted BayeENproteomics object from 'output_name' folder made during its creation
    def load(self):
        UniProt = pd.read_table(self.output_name+'\\UniProt.tab',sep = '\t')
        peptides_used = pd.read_table(self.output_name+'\\peptides_used.tab',sep = '\t')
        input_table = pd.read_table(self.output_name+'\\input_table.csv',sep = ',')
        longtable = pd.read_table(self.output_name+'\\longtable.tab',sep = '\t')
        self.UniProt = UniProt
        self.peptides_used = peptides_used
        self.input_table = input_table
        self.longtable = longtable
        try:
            missing_peptides_idx = pd.read_table(self.output_name+'\\missing_peptides_idx.csv',sep = ',')
            self.missing_peptides_idx = missing_peptides_idx
        except:
            print('missing_peptides_idx output missing')

        try:
            allValues = pd.read_table(self.output_name+'\\allValues.csv',sep = ',')
            self.allValues = allValues
        except:
            print('allValues output missing')

        try:
            protein_summary_quant = pd.read_table(self.output_name+'\\protein_summary_quant.csv',sep = ',')
            self.protein_summary_quant = protein_summary_quant
        except:
            print('protein_summary_quant output missing')

        try:
            protein_subject_quant = pd.read_table(self.output_name+'\\protein_subject_quant.csv',sep = ',')
            self.protein_subject_quant = protein_subject_quant
        except:
            print('protein_subject_quant output missing')

        try:
            PTM_summary_quant = pd.read_table(self.output_name+'\\PTM_summary_quant.csv',sep = ',')
            self.PTM_summary_quant = PTM_summary_quant
        except:
            print('PTM_summary_quant output missisng')

        try:
            isoform_summary_quant = pd.read_table(self.output_name+'\\isoform_summary_quant.csv',sep = ',')
            self.isoform_summary_quant = isoform_summary_quant
        except:
            print('isoform_summary_quant output missing')

        try:
            PTM_subject_quant = pd.read_table(self.output_name+'\\PTM_subject_quant.csv',sep = ',')
            self.PTM_subject_quant = PTM_subject_quant
        except:
            print('PTM_subject_quant output missing')

        try:
            pathway_summary_quant = pd.read_table(self.output_name+'\\pathway_summary_quant.csv',sep = ',')
            self.pathway_summary_quant = pathway_summary_quant
        except:
            print('pathway_summary_quant output missing')

        try:
            other_summary_quant = pd.read_table(self.output_name+'\\other_summary_quant.csv',sep = ',')
            self.other_summary_quant = other_summary_quant
        except:
            print('other_summary_quant output missing')

        try:
            Reactome = pd.read_table(self.output_name+'\\Reactome.csv',sep = ',')
            self.Reactome = Reactome
        except:
            print('Reactome pathway data missing')

    # Compare all proteins between two experimental groups, repeated calls overwrite previous self.Constrasted
    def doContrasts(self, Contrasted = 'protein', ctrl=0, propagateErrors=False, UseBayesFactors=False, continuous=False):

        if Contrasted == 'protein':
            self.Contrasted = dc(self.protein_summary_quant)
        elif Contrasted == 'ptm':
            self.Contrasted = dc(self.PTM_summary_quant)
        elif Contrasted == 'pathway':
            self.Contrasted = dc(self.pathway_summary_quant)
        elif Contrasted == 'other':
            self.Contrasted = dc(self.other_summary_quant)
        else:
            msg = 'Contrasted must be ''pathway'', ''protein'', ''ptm'' or ''other''.'
            raise InputError(Contrasted,msg)

        DoFs = np.array(self.Contrasted['degrees of freedom'])[:,np.newaxis].astype(float)
        FCcolumns = np.array(self.Contrasted.iloc[:,['fold change}' in i for i in self.Contrasted.columns]])
        ctrlvals = FCcolumns[:,ctrl][:,np.newaxis]
        if continuous:
            FCs = FCcolumns
        else:
            FCs = FCcolumns - ctrlvals
        nProteins, nGroups = FCcolumns.shape
        SEs = np.array(self.Contrasted.iloc[:,['{SE}' in i for i in self.Contrasted.columns]])

        if propagateErrors: #Equivalent to doing a Welch's t-test for when variances are unequal
            SEs[:,ctrl+1:] = np.sqrt(SEs[:,ctrl+1:]**2 + SEs[:,ctrl][:,np.newaxis]**2)
            if ctrl:
                SEs[:,0:ctrl] = np.sqrt(SEs[:,0:ctrl]**2 + SEs[:,ctrl][:,np.newaxis]**2)

        if UseBayesFactors:
            BF = FCs**2/(2*SEs**2)
            self.Contrasted.iloc[:,['{EB t-test p-value}' in i for i in self.Contrasted.columns]] = BF
        else:
            t = abs(FCs)/SEs
            pvals = np.minimum(1,stats.t.sf(t, DoFs - 1) * 2)
            fdradjp = dc(pvals)

            for i in range(nGroups):
                fdradjp[:,i] = bhfdr(pvals[:,i])

            self.Contrasted.iloc[:,['{EB t-test p-value}' in i for i in self.Contrasted.columns]] = pvals
            self.Contrasted.iloc[:,['{BHFDR}' in i for i in self.Contrasted.columns]] = fdradjp

        self.Contrasted.iloc[:,['fold change}' in i for i in self.Contrasted.columns]] = FCs
        self.Contrasted.iloc[:,['{SE}' in i for i in self.Contrasted.columns]] = SEs
        self.Contrasted.to_csv(self.output_name+'\\Contrasted_vs_Col'+str(ctrl)+'.csv', encoding='utf-8', index=False,header=self.Contrasted.columns)

    def boxplots(self):
        # Use for fold-change outlier detection
        nG = np.array(self.input_table['group number'])[0]
        fig1, (ax1,ax2,ax3,ax4) = mpl.subplots(1,4, figsize=(nG*5, 4), sharex=True)
        ax1.boxplot(self.protein_summary_quant.iloc[:,4:4+nG].T,notch=True,labels=self.protein_summary_quant.columns[4:4+nG])
        ax1.set_title('Protein effect sizes')
        ax1.set_ylabel(r'$Log_2 (abundance)$')
        ax2.boxplot(self.PTM_summary_quant.iloc[:,9:9+nG].T,notch=True,labels=self.PTM_summary_quant.columns[9:9+nG])
        ax2.set_title('PTM effect sizes')
        ax3.boxplot(self.isoform_summary_quant.iloc[:,4:4+nG].T,notch=True,labels=self.isoform_summary_quant.columns[4:4+nG])
        ax3.set_title('Unmodified peptide effect sizes')
        #ax2.set_ylabel(r'$Log_2 (condition / ctrl)$')
        ax4.boxplot(self.pathway_summary_quant.iloc[:,5:5+nG].T,notch=True,labels=self.pathway_summary_quant.columns[5:5+nG])
        ax4.set_title('Pathway effect size')
        mpl.show()
        #ax3.set_ylabel(r'$Log_2 (condition / ctrl)$')

    def volcanoes(self,plot_type = 'protein', residue = 'any'):
        nG = np.array(self.input_table['group number'])[0]
        fig, axes = mpl.subplots(1,nG, figsize=(nG*5, 4),squeeze=False)

        for plot in range(nG):
            if plot_type == 'protein':
                #Protein plots
                axes[0,plot].scatter(self.Contrasted.iloc[:,4+plot], -np.log10(self.Contrasted.iloc[:,4+3*nG+plot]))
                axes[0,plot].set_xlabel(self.Contrasted.columns[4+plot])
                axes[0,plot].axis('tight')

            elif plot_type == 'ptm':
                #PTM plots
                axes[0,plot].scatter(self.Contrasted.iloc[:,9+plot], -np.log10(self.Contrasted.iloc[:,9+3*nG+plot]))
                axes[0,plot].set_xlabel(self.Contrasted.columns[9+plot])
                axes[0,plot].axis('tight')

            elif plot_type == 'pathway':
                #Pathway plots
                axes[0,plot].scatter(self.Contrasted.iloc[:,5+plot], -np.log10(self.Contrasted.iloc[:,5+3*nG+plot]))
                axes[0,plot].set_xlabel(self.Contrasted.columns[5+plot])
                axes[0,plot].axis('tight')

            else:
                #PTM-specific volcanoes
                try:
                    o = self.Contrasted[self.Contrasted['PTM type'] == plot_type]
                except:
                    print(self.output_name+'.Contrasted does not have column "PTM type". Probably need to redo '+self.output_name+'.doContrasts using "ptm" as an arguement.' )

                if residue != 'any':
                     o = o[o['PTMed residue'] == residue]

                axes[0,plot].scatter(o.iloc[:,9+plot], -np.log10(o.iloc[:,9+3*nG+plot]))
                axes[0,plot].set_xlabel(self.Contrasted.columns[9+plot])
                axes[0,plot].axis('tight')

            axes[0,plot].plot(np.array([-100, 100]),np.array([-np.log10(0.05),-np.log10(0.05)]),color='black',linestyle='dashed')
            axes[0,plot].set_ylabel(r'${-Log_1}_0 (BHFDR)$')
            axes[0,plot].set_ylim((0,100))

    def pca(self,plot_type = 'protein', residue = 'any', dim = 2, annotate = True):

        if dim < 2:
            dim=2
        elif dim > 3:
            dim = 3

        nRuns = np.array(self.input_table['run number'])[0]

        if plot_type == 'protein':
            o = self.protein_subject_quant.iloc[:,0:nRuns].transpose()
        elif plot_type == 'peptide':
            o = self.allValues.iloc[1:,2:].transpose()
            #missingidx = np.array(self.missing_peptides_idx.iloc[1:,:].transpose())==True
            #o[missingidx] = np.nan
        elif plot_type == 'ptm':
            o = self.PTM_subject_quant.iloc[:,0:nRuns].transpose()
        else:
            #PTM-specific PCA
            o = self.PTM_subject_quant[self.PTM_summary_quant['PTM type'] == plot_type]
            if residue != 'any':
                o = o[self.PTM_summary_quant[self.PTM_summary_quant['PTM type'] == plot_type]['PTMed residue'] == residue]

        o_zscored = stats.zscore(o, axis=0)

        nG = np.array(self.input_table['group number'])[0]
        groups = np.unique(self.longtable['Treatment'])

        pca = PCA(n_components=3)
        pca.fit(o_zscored)
        X_pca = pca.transform(o_zscored)

        fig = mpl.figure()

        if dim > 2:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        for i in range(nG):
            group_finder = np.where(o.index.str.contains(groups[i],regex=True))
            if dim > 2:
                ax.scatter(X_pca[group_finder, 0], X_pca[group_finder, 1], X_pca[group_finder, 2], label = groups[i])
            else:
                ax.scatter(X_pca[group_finder, 0], X_pca[group_finder, 1], label = groups[i])

        ax.legend(loc='best', shadow=False, scatterpoints=1)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        if annotate:
            for i in range(len(o.index)):
                if dim > 2:
                    ax.text(X_pca[i,0], X_pca[i,1], X_pca[i,2], o.index[i])
                else:
                    ax.text(X_pca[i,0], X_pca[i,1], o.index[i])

        if dim > 2:
            ax.set_zlabel('PC3')

            # rotate the axes and update
            for angle in range(0, 360):
                ax.view_init(30, angle)
                mpl.draw()
                mpl.pause(.001)

        mpl.show()

        fig.savefig(self.output_name+'\\'+str(dim)+'D-PCA_type='+plot_type+residue+'.pdf', bbox_inches='tight')

    def proteinVSptm(self, ptm_type, residue = 'any', exp = 1, ctrl = 0, continuous = False):
        o = self.PTM_summary_quant[self.PTM_summary_quant['PTM type'] == ptm_type]
        if residue != 'any':
            o = o[o['PTMed residue'] == residue]

        PTMfoldchanges = np.array(o.iloc[:,9+exp]) - np.array(o.iloc[:,9+ctrl])
        proteinfoldchanges = dc(PTMfoldchanges)

        for i,j in zip(o['Parent protein'],range(len(o['Parent protein']))):
            if continuous:
                proteinfoldchanges[j] = np.array(self.protein_summary_quant[self.protein_summary_quant['Protein'] == i].iloc[:,4+exp])
            else:
                proteinfoldchanges[j] = np.array(self.protein_summary_quant[self.protein_summary_quant['Protein'] == i].iloc[:,4+exp]) - np.array(self.protein_summary_quant[self.protein_summary_quant['Protein'] == i].iloc[:,4+ctrl])

        PTMProt_table = pd.DataFrame({'PTM log2(fold changes)':PTMfoldchanges,'Protein log2(fold changes)':proteinfoldchanges})
        print(PTMProt_table)
        PTMProt_table.to_csv(self.output_name+'\\PTMvsProtein_.csv', encoding='utf-8', index=False,header=PTMProt_table.columns)

        mpl.scatter(proteinfoldchanges, PTMfoldchanges)
        mpl.gca().set_xlabel(self.protein_summary_quant.columns[4+exp]+'protein')
        mpl.gca().set_ylabel(self.PTM_summary_quant.columns[9+exp]+ptm_type)

    def proteoformDiscoverer(self, protein, dim = 2):
        #Look at interaction coefficients (treatment*peptide betas) for all peptides belonging to a given protein to see which deviate from consensus profile (treatment beta).
        #Cluster them based on how they respond to different treatments to gain insight into how many physiologically relevant proteoforms are in a dataset.
        nG = np.array(self.input_table['group number'])[0]
        protein_PTMs = self.PTM_summary_quant[self.PTM_summary_quant['Parent protein'] == protein]
        protein_isos = self.isoform_summary_quant[self.isoform_summary_quant['Parent Protein'] == protein]

        protein_PTMs_values = protein_PTMs.iloc[:,9:9+nG]
        protein_isos_values = protein_isos.iloc[:,4:4+nG]

        peptides = pd.concat([protein_PTMs['Peptide'], protein_isos['Peptide']], sort=False).reset_index(drop=False)
        values = pd.concat([protein_PTMs_values, protein_isos_values], sort=False).reset_index(drop=True)

        proteoform_values = np.array(values)

        if dim < 2:
            dim=2
        elif dim > 3:
            dim = 3


        #proteoform_values = stats.zscore(proteoform_values, axis=1)
        proteoform_values = proteoform_values - np.mean(proteoform_values,axis=1)[:,np.newaxis]

        # Hierarchical clustering based on proteoform values
        linked = linkage(proteoform_values, 'single')
        #labelList = list(range(1, proteoform_values.shape[0]))
        mpl.figure(figsize=(10, 7))
        dendrogram(linked,
            orientation='top',
            #labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
        #mpl.xlabel(peptides['Peptide'])
        mpl.ylabel('Distance')
        mpl.show()

        # Use JumpMethod to calculate ideal number of k-means clusters, get cluster assignments.
        n_clusters = JumpMethod(proteoform_values)
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(proteoform_values)

        # PCA showing different clusters
        pca = PCA(n_components=3)
        pca.fit(proteoform_values)
        X_pca = pca.transform(proteoform_values)

        fig = mpl.figure(figsize=(10, 7))

        if dim > 2:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        for i in np.unique(cluster_labels):
            if dim > 2:
                ax.scatter(X_pca[cluster_labels == i, 0],
                           X_pca[cluster_labels == i, 1],
                           X_pca[cluster_labels == i, 2],
                           label = 'Cluster '+str(i))
                        #label = peptides['Peptide'].loc[cluster_labels == i].astype(str).sum())
            else:
                ax.scatter(X_pca[cluster_labels == i, 0],
                           X_pca[cluster_labels == i, 1],
                           label = 'Cluster '+str(i))

        ax.legend(loc='best', shadow=False, scatterpoints=1)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        if dim > 2:
            ax.set_zlabel('PC3')

            # rotate the axes and update
            for angle in range(0, 360):
                ax.view_init(30, angle)
                mpl.draw()
                mpl.pause(.001)

        mpl.show()

        cluster_assignments = pd.DataFrame({'Cluster assignemnt':cluster_labels})
        proteoform_table = pd.concat([peptides,values,cluster_assignments],axis=1,sort=False)

        proteoform_table.to_csv(self.output_name+'\\'+protein+'_Proteoforms.csv',encoding='utf-8', index=False,header=proteoform_table.columns)


# Wrapper for pathway model fitting
def fitPathwayModels(models,uniprotall,species,model_table,nRuns,isPTMfile=False,subjectQuant=[],download = True,nChains=3,continuousvars=[]):

    try:
        if not download:
            uniprot2reactomedf = pd.read_table("UniProt2Reactome.txt",header=None)
        else:
            msg = 'UniProt2Reactome.txt not found. Attempting to download from Reactome.org'
            raise InputError(download,msg)
    except:
        print('Getting Reactome Pathway data...')
        url = 'http://reactome.org/download/current/UniProt2Reactome.txt'
        request = urllib.request.Request(url)
        response = urllib.request.urlopen(request)
        uniprot2reactome = response.read().decode('utf-8')

        with open("UniProt2Reactome.txt", "w",encoding="utf-8") as txt:
            print(uniprot2reactome, file = txt)

    if species == 'human':
        species = 'Homo sapiens'
    elif species == 'mouse':
        species = 'Mus musculus'
    else:
        url = 'https://www.uniprot.org/proteomes/'+species
        request = urllib.request.Request(url)
        response = urllib.request.urlopen(request)
        species = response.read().decode('utf-8')
        species = re.search(r'<title>(\w+ \w+) \(',species)[1]

    uniprot2reactomedf = pd.read_table("UniProt2Reactome.txt",header=None)
    uniprot2reactomedf = uniprot2reactomedf.loc[uniprot2reactomedf.iloc[:,-1].isin([species]),:].reset_index(drop=True)
    print('Got Reactome pathway data')

    # Annotate pathway file with Identifiers used in protein quant
    u2r_protein_id = pd.DataFrame({'Entry name':['']*uniprot2reactomedf.shape[0]}) #columns = ['Entry name'])
    for uniprotID in list(uniprotall.iloc[:,0].values.flatten()):
        u2r_protein_finder = uniprot2reactomedf.iloc[:,0].isin([uniprotID])
        protein_list_finder = models['Entry'].isin([uniprotID])
        if np.any(u2r_protein_finder) and np.any(protein_list_finder):
            name = uniprotall.loc[uniprotall.iloc[:,0].isin([uniprotID])]['Entry name'].iloc[0]
            u2r_protein_id.loc[u2r_protein_finder] = name
            print('Pathways found for', name)
    uniprot2reactomedf['Entry name'] = u2r_protein_id.reset_index(drop=True)

    # Get pathways represented in models and assign proteins in models to pathways
    x1 = uniprot2reactomedf.shape[0]
    x2 = models.shape[0]
    PathwayCount = np.zeros((x1,1))
    ProteinsInPathway = np.zeros((x2,x1))
    for i in range(x2):
        protein_finder = uniprot2reactomedf['Entry name'].isin([models.iloc[i,0]])
        PathwayCount = PathwayCount + protein_finder[:,np.newaxis].astype(int)
        ProteinsInPathway[i,:] = ProteinsInPathway[i,:] + protein_finder.T
        print('Protein #',i, models.iloc[i,0],' found in ', np.sum(protein_finder), ' pathway(s).')
    uniprot2reactomedf = uniprot2reactomedf.loc[(PathwayCount > 0).flatten(),:]
    ProteinsInPathway = ProteinsInPathway[:,(PathwayCount > 0).flatten()]

    uniquePathways,ic,ia,totalPinP = np.unique(uniprot2reactomedf.iloc[:,1],return_index=True,return_inverse=True,return_counts=True)
    uniprot2reactomedf2 = uniprot2reactomedf.iloc[ic,:]
    ProteinsInPathway2 = np.zeros((x2,uniquePathways.shape[0]))
    #totalPinP = np.zeros((uniquePathways.shape[0],1))

    for i in range(uniquePathways.size):
        ProteinsInPathway2[:,i] = np.sum(ProteinsInPathway[:,ia==i],axis=1)
    uniprot2reactomedf2 = uniprot2reactomedf2.loc[(totalPinP >= 5).flatten(),:]
    x1 = uniprot2reactomedf2.shape[0]
    if x1 > 1:
        ProteinsInPathway2 = ProteinsInPathway2[:,(totalPinP >= 5).flatten()]
    else:
        ProteinsInPathway2 = ProteinsInPathway2[:,(totalPinP >= 5).flatten()][:,np.newaxis]
    FCcolumns = models.loc[:,['fold change}' in i for i in models.columns]]
    nGroups = FCcolumns.shape[1]

    '''
    if not subjectQuant.empty:
        a = int(nRuns/nGroups)
        FCcolumns = np.array(subjectQuant.iloc[:,0:nRuns]).astype(float)
        SEcolumns = np.tile(np.array(models.loc[:,['{SE}' in i for i in models.columns]]),(1,a)).astype(float)
        doWeights = False
    else:
        a = 1
        SEcolumns = np.array(models.loc[:,['{SE}' in i for i in models.columns]]).astype(float)
        doWeights = True
    '''
    a = 1
    SEcolumns = np.array(models.loc[:,['{SE}' in i for i in models.columns]]).astype(float)
    doWeights = True
    if 'Group' in continuousvars:
        t1 = ['Group']
    else:
        t1 = list(np.unique(model_table['Group']))
    #print(continuousvars,t1)
    column_names = quantTableNameConstructor(t1,nRuns,isSubjectLevelQuant = False)
    PathwayQuant = pd.DataFrame(columns = ['Pathway ID','Pathway description', '# proteins','degrees of freedom','MSE','protein list']+column_names)
    pathway_models = {}
    for i in range(x1):
        start = time.time()
        if isPTMfile:
            proteins = np.tile(np.array(models.loc[(ProteinsInPathway2[:,i] > 0).flatten(),models.columns[['Parent protein','PTM position in protein']]].astype(str).sum(axis=1)),(nGroups*a,1)).flatten()
            nsites = np.unique(proteins).size
            nprot = models['Parent protein'].loc[(ProteinsInPathway2[:,i] > 0).flatten()].shape[0]
            nprot2 = np.unique(nprot).size
        else:
            proteins = np.tile(np.array(models['Protein'].loc[(ProteinsInPathway2[:,i] > 0).flatten()]),(nGroups*a,1)).flatten()
            nprot = models['Protein'].loc[(ProteinsInPathway2[:,i] > 0).flatten()].shape[0]
            nprot2 = nprot
            nsites = nprot

        if nsites < 5 or nprot2 < 5:
            continue

        abundances = np.array(FCcolumns)[(ProteinsInPathway2[:,i] > 0).flatten(),:].flatten()
        treatments = np.tile(t1,(a*nprot,1)).flatten()
        SEs = np.minimum(1,abs(abundances/np.array(SEcolumns[(ProteinsInPathway2[:,i] > 0).flatten(),:]).flatten()))[:,np.newaxis]
        pathway_table = pd.DataFrame({'Protein':proteins,
                                      'Treatment':treatments})

        X = pd.get_dummies(pathway_table)
        parameterIDs = X.columns
        X = np.array(X,dtype=int)
        Y = np.array(abundances)[:,np.newaxis]
        n = X.shape[0]

        # Fit model
        pathwaymdl = weighted_bayeslm_multi(X,Y,parameterIDs,doWeights,SEs,np.array([]),0,[],nChains)
        results = pathwaymdl['beta_estimate']
        SEMs = pathwaymdl['SEMs']
        dof = n-1
        #if not subjectQuant.empty:
        #    dof = pathwaymdl['dof']
        #else:
        #    dof = n-1#pathwaymdl['dof']

        Treatment_i = effectFinder(parameterIDs,'Treatment')
        Treatment_betas = list(results[Treatment_i])
        Treatment_SEMs = list(SEMs[Treatment_i])
        PathwayQuant = PathwayQuant.append(dict(zip(['Pathway ID','Pathway description', '# proteins','degrees of freedom','MSE','protein list']+column_names,[uniprot2reactomedf2.iloc[i,1],uniprot2reactomedf2.iloc[i,3],nprot2,dof,pathwaymdl['residVar'],list(np.unique(proteins))]+Treatment_betas+Treatment_SEMs+[1]*nGroups*2)),ignore_index=True,sort=False) #We'll calculate p-values (Bayes Factors?) and FDR-adjusted p-values later on.

        pathway_models[uniprot2reactomedf2.iloc[i,3]] = pathwaymdl

        timetaken = time.time()-start
        print('#',i,'/',x1,uniprot2reactomedf2.iloc[i,1],uniprot2reactomedf2.iloc[i,3],nprot2,dof,Treatment_betas, 'Took {:.2f} minutes.'.format(timetaken/60))

    if PathwayQuant.shape[0] < 1:
        print('0 pathways with at least 5 proteins found!')

    return PathwayQuant, pathway_models, uniprot2reactomedf

# Modify SE estimates and degrees of freedom as per Empirical Bayes (Smyth 2004)
def EBvar(models):

    SEcolumns = np.array(models.iloc[:,['{SE}' in i for i in models.columns]])
    nProteins, nGroups = SEcolumns.shape
    #d0 = [0]*nGroups
    #s0 = [0]*nGroups
    DoFs = np.array(models['degrees of freedom'])
    EBDoFs = dc(DoFs)

    if nProteins < 2:
        d0 = 0
        s0 = 0
    else:
        d0, null, s0 = stats.chi2.fit(SEcolumns.flatten())

    EBDoFs = DoFs + d0
    for i in range(nGroups):
        for ii in range(nProteins):
            lam = DoFs[ii]/(DoFs[ii]+d0)
            oldVar = (SEcolumns[ii,i]/np.sqrt(2/((DoFs[ii]/2) + 2)))**2
            newSEM = np.sqrt(lam * oldVar + (1 - lam) * s0) * np.sqrt(2/((EBDoFs[ii])/2 + 2))
            SEcolumns[ii,i] = newSEM

    models.iloc[:,['{SE}' in i for i in models.columns]] = SEcolumns
    models['degrees of freedom'] = EBDoFs

    return models,{'d0':d0, 's0':s0}

# Data-wrangler function
def formatData(normpeplist,
                exppeplist,
                organism,
                othermains_bysample = '',
                othermains_bypeptide = '',
                normmethod='median',
                ProteinGrouping=False,
                scorethreshold=0.2,
                nDB=1,
                regression_method = 'protein',
                ContGroup=[],
                form='progenesis',
                download=True,
                impute='ami',
                reassign_unreviewed=True,
                useReviewedOnly=False):

    #get uniprot info
    print('Getting Uniprot data for',organism)
    uniprotall, upcol = getUniprotdata(organism,download)
    uniprotall_reviewed = uniprotall.loc[uniprotall.iloc[:,-2] == 'reviewed',:]

    if useReviewedOnly:
        uniprotall=uniprotall_reviewed

    print('Done!')

    #import peptide lists
    if form == 'progenesis':
        e_peplist,GroupNames,runIDs,RA,nEntries,nRuns = Progenesis2BENP(exppeplist)
    elif form == 'maxquant':
        e_peplist,GroupNames,runIDs,RA,nEntries,nRuns = MaxQuant2BENP(exppeplist)
    elif form == 'peaks':
        e_peplist,GroupNames,runIDs,RA,nEntries,nRuns = PEAKS2BENP(exppeplist)
    else:
        msg = 'form must be ''progenesis'' (default), ''peaks'' or ''maxquant''.'
        raise InputError(form,msg)

    if form == 'maxquant' or form == 'peaks':
        #Convert MaxQuant/PEAKS protein names to UniProt Accession codes
        accessions = pd.DataFrame(data = ['']*e_peplist.shape[0], columns = ['Accession'])
        for n,name in zip(e_peplist.index,e_peplist['Accession']):
            #print(n,uniprotID)
            try:
                IDs = name.split(';')
                accession = ''
                for i in IDs:
                    #IDfinder = uniprotall.iloc[:,1].isin([i.split('|')[-1]])
                    accession = accession + i.split('|')[-1] + ';'#uniprotall.loc[IDfinder,uniprotall.columns[1]]
            except:
                accession = 'N/A'
            accessions.iloc[n] = accession
            print(n, accession)
        e_peplist['Accession'] = accessions

    if normpeplist != exppeplist:
        #print('Importing peptide lists:', normpeplist, exppeplist)
        if form == 'progenesis':
            n_peplist = Progenesis2BENP(normpeplist)[0]
        elif form == 'maxquant':
            n_peplist = MaxQuant2BENP([normpeplist])[0]
        elif form == 'peaks':
            n_peplist = PEAKS2BENP(normpeplist)[0]
    else:
        n_peplist = dc(e_peplist)

    # Handle specification for continuous treatment variable
    if ContGroup:
        GroupIDs = set(['Continuous variable'])
        nGroups = 2
    else:
        if form == 'maxquant' or form == 'peaks':
            for i in range(len(GroupNames)):
                GroupNames[i] = re.sub(r'(\.[0-9]+)','',GroupNames[i])

        GroupIDs = set(GroupNames)
        nGroups = len(GroupIDs)
        print(GroupIDs)

    if othermains_bysample != '':
        e_mainsampleeffects = pd.read_csv(othermains_bysample) #.csv assigning additional column varibles to each run
        #print(e_mainsampleeffects)
        try:
            GroupNames = e_mainsampleeffects['Group']
            GroupIDs = set(GroupNames)
            nGroups = len(GroupIDs)
            e_mainsampleeffects = e_mainsampleeffects.loc[:,e_mainsampleeffects.columns != 'Group']
        except:
            if form == 'progenesis':
                print('No "Group" variable in '+othermains_bysample+'. Using column headers from '+exppeplist+'.')
            else:
                print('No "Group" variable in '+othermains_bysample+'. Using column headers from '+exppeplist[0]+'.')
        nSmains = e_mainsampleeffects.shape[1]
        othermains_bysample_names = e_mainsampleeffects.columns
    if othermains_bypeptide != '':
        e_mainpeptideeffects = pd.read_csv(othermains_bypeptide) #.csv assigning additional row varibles to each peptide
        othermains_bypeptide_names = e_mainpeptideeffects.columns
        nPmains = e_mainpeptideeffects.shape[1]

    #header = e_peplist.iloc[0:1,:];
    e_length = e_peplist.shape[0]
    scores = e_peplist['Score']
    scores[scores == '---'] = np.nan
    scorebf = (np.log10(1/(20*e_length)) * -10) - 13
    pepID_p = 10**(np.array(scores, dtype=float)/-10)
    if form == 'progenesis':
        #e_length = e_peplist.shape[0]
        scores = np.array(e_peplist.iloc[2:,7], dtype = object)
        scores[scores == '---'] = np.nan
        #print(np.array(scores, dtype=float))
        scorebf = (np.log10(1/(20*(e_length - 2))) * -10) - 13
        pepID_p = 10**(np.array(scores, dtype=float)/-10) #Transform Mascot scores back to p-values
    elif form == 'maxquant':
        pepID_p = np.array(scores, dtype=float)
        e_peplist['Score'] = (np.log10(e_peplist['Score']) * -10)

    #print(scorebf)

    pepID_fdr = bhfdr(pepID_p)
    #print(pepID_fdr)
    #e_peplist = pd.concat([header,e_peplist.iloc[3:,:].loc[pepID_fdr < scorethreshold,:]])
    if othermains_bypeptide != '':
        e_mainpeptideeffects = pd.concat([e_mainpeptideeffects,e_peplist['Accession']],axis=1)
        e_mainpeptideeffects = e_mainpeptideeffects.loc[pepID_fdr < scorethreshold,:]
        e_mainpeptideeffects = e_mainpeptideeffects.sort_values(by=['Accession'])

    # Initial check of reviewed status of each protein in dataset
    # Create unique (sequence and mods) id for each peptide
    if form == 'progenesis':
        e_peplist = e_peplist.iloc[2:,:].loc[pepID_fdr < scorethreshold,:]
    else:
        e_peplist = e_peplist.loc[pepID_fdr < scorethreshold,:]

        #for i in range(e_peplist.shape[0]):
        #    e_peplist.iloc[i,10] = e_peplist.iloc[i,10].split(';')[0].split('|')[-1]

    e_peplist = e_peplist.sort_values(by=['Accession'])
    e_length = e_peplist.shape[0]
    gene_col = 6
    ii = 0

    while ii != e_length:   #For future, make import formats uniform!

        protein_assc = e_peplist.iloc[ii,10]
        if protein_assc == '':
            ii = ii + 1
            continue

        protein_find = e_peplist.iloc[:,10].isin([protein_assc])

        #protein_find.mask(protein_find == 0)
        if nDB > 1:
            protein_name = protein_assc[3:]
        else:
            protein_name = protein_assc
        if form == 'maxquant' or form == 'peaks':
            protein_name = protein_name[0:-1]

        uniprot_find = uniprotall.iloc[:,1].isin([protein_name])
        #uniprot_find.mask(unprot_find == 0)
        review_status = uniprotall['Status'].loc[uniprot_find]
        protein_id = uniprotall.loc[uniprot_find,uniprotall.columns[[upcol,1]]]
        protein_sequence = uniprotall['Sequence'].loc[uniprot_find]
        #e_peplist.loc[protein_find,e_peplist.columns[1]] = e_peplist.loc[protein_find,e_peplist.columns[[8,9]]].astype(str).sum(axis=1)
        e_peplist['PeptideID'].loc[protein_find] = e_peplist[['Sequence','Modifications']].astype(str).sum(axis=1)
        #e_peplist.loc[protein_find,e_peplist.columns[2]] = e_peplist.loc[protein_find,e_peplist.columns[gene_col+int(not ProteinGrouping)*4]]
        e_peplist['Charge'].loc[protein_find] = e_peplist.loc[protein_find,e_peplist.columns[gene_col+int(not ProteinGrouping)*4]]
        if protein_sequence.shape[0] > 0:
            #e_peplist.loc[protein_find,e_peplist.columns[5]] = np.tile(protein_sequence,(int(np.sum(protein_find)),1)).flatten()
            e_peplist['ProteinSequence'].loc[protein_find] = np.tile(protein_sequence,(int(np.sum(protein_find)),1)).flatten()
        else:
            #e_peplist.loc[protein_find,e_peplist.columns[5]] = np.tile([''],(int(np.sum(protein_find)),1)).flatten()
            e_peplist['ProteinSequence'].loc[protein_find] = np.tile([''],(int(np.sum(protein_find)),1)).flatten()

        #if len(protein_id.index):
        try:
            e_peplist.loc[protein_find,e_peplist.columns[[gene_col,10]]] = np.tile(protein_id.iloc[0,:],(int(np.sum(protein_find)),1))
            review_status = 'reviewed'
        #else:
        except:
            e_peplist.loc[protein_find,e_peplist.columns[[gene_col,10]]] = protein_name
            review_status = 'unreviewed'

        e_peplist['Reviewed?'].iloc[ii:] = review_status
        print('#',ii,'-',ii+int(np.sum(protein_find)),'/',e_length, protein_name, review_status)
        ii = ii + int(np.sum(protein_find))

    #e_peplist = pd.concat([header,e_peplist.iloc[2:,:].sort_values(by=['Unnamed: 7'])])
    e_peplist = e_peplist.sort_values(by=['Score'])

    # Do normalisation to median intensities of specified peptides
    if normmethod == 'median' or normmethod == 'quantile':
        e_peplist.iloc[:,RA:] = np.log2(e_peplist.iloc[:,RA:].astype(float))
        if form == 'progenesis':
            norm_intensities = np.log2(n_peplist.iloc[2:,RA:].astype(float))
        else:
            norm_intensities = np.log2(n_peplist.iloc[:,RA:].astype(float))
        normed_peps = normalise(e_peplist.iloc[:,RA:].astype(float),norm_intensities,normmethod)
        normed_peps[np.isinf(normed_peps)] = np.nan
    elif normmethod == 'none':
        print('No normalisation used!')
        normed_peps = np.log2(e_peplist.iloc[:,RA:].astype(float))
        normed_peps[np.isinf(normed_peps)] = np.nan
    e_peplist.iloc[:,RA:] = np.array(normed_peps)

    #Do imputation
    if impute != 'ami':
        e_peplist = Impute(e_peplist, impute, RA)
        normed_peps = e_peplist.iloc[:,RA:]
        #e_peplist.to_csv('X.csv', encoding='utf-8', index=False,header=e_peplist.columns)

    unique_pep_ids = np.unique(e_peplist.iloc[:,8])
    nUniquePeps = len(unique_pep_ids)
    q = 0
    final_peplist = dc(e_peplist)
    if othermains_bypeptide != '':
        final_pepmaineffectlist = dc(e_mainpeptideeffects)
    print(nUniquePeps, 'unique (by sequence) peptides.')

    # Setup protein matrix if whole dataset regression is to be performed
    if regression_method == 'dataset':
        #print(pd.DataFrame(np.tile('Protein_',(uniprotall.shape[0],1))).shape,uniprotall.iloc[:,0][:,np.newaxis].shape)
        protein_columns = pd.concat((pd.DataFrame(np.tile('Protein_',(uniprotall.shape[0],1))),uniprotall['Entry name']),axis=1).astype(str).sum(axis=1)
        protein_matrix = pd.DataFrame(np.zeros((final_peplist.shape[0],uniprotall.shape[0])),columns = protein_columns)

    # Loop to reassign peptides from unreviewed proteins to reviewed ones if sequences match
    # prefers reviewed proteins that are already in the dataset
    for i in unique_pep_ids:
        pep_find = np.array(e_peplist.iloc[:,8].isin([i])) #find all instances of a peptide
        #print(np.where(pep_find)[0])
        #return
        pep_info = dc(e_peplist.loc[pep_find,:])
        pep_values = normed_peps.loc[pep_find,:]
        if othermains_bypeptide != '':
            #print(pep_find)
            #print(e_mainpeptideeffects.head())
            pep_effects = dc(e_mainpeptideeffects.loc[pep_find,:])
        nP = pep_info.shape[0]

        # Fill protein matrix
        if regression_method == 'dataset':
            pep2prot = uniprotall['Sequence'].str.contains(pep_info.iloc[0,8]).astype(int) #Find all proteins that peptide could be part of
            protein_matrix.loc[pep_find,:] = np.tile(pep2prot.T,(nP,1))
            continue

        #print(pep_values.shape,q,q-1+nP)
        if np.any(pep_info.iloc[:,0].isin(['unreviewed'])) and reassign_unreviewed:
            reviewed_prot_find = uniprotall_reviewed['Sequence'].str.contains(pep_info.iloc[0,8]) #Find all reviewed proteins that peptide could be part of
            reviewed_prot_findidx = np.where(reviewed_prot_find)[0]
            if reviewed_prot_findidx.any():
                new_protein_ids = uniprotall_reviewed.loc[reviewed_prot_find,uniprotall_reviewed.columns[[1+int(ProteinGrouping)*10,upcol]]]
                new_protein_seq = uniprotall_reviewed.loc[reviewed_prot_find,uniprotall_reviewed.columns[12]]
                is_present = np.zeros([e_length,new_protein_ids.shape[0]])
                for ii in range(new_protein_ids.shape[0]):
                    #Reassign to proteins already in dataset if possible
                    is_present[:,ii] = e_peplist.iloc[:,gene_col+int(not ProteinGrouping)*4].isin([new_protein_ids.iloc[ii,int(ProteinGrouping)]])
                is_present = np.sum(is_present,axis=0)
                ia = np.argmax(is_present)
                ic = np.amax(is_present)
                #If peptide could be shared between 2 or more equally likely reviewed proteins, skip reassignment.
                if ic != np.amin(is_present) or is_present.size == 1:
                    #print(pep_info.loc[:,pep_info.columns[[10,6]]],new_protein_ids.iloc[ia,:])
                    pep_info.loc[:,pep_info.columns[[10,6]]] = np.array(np.tile(new_protein_ids.iloc[ia,:],(nP,1)))
                    pep_info['Reviewed?'] = ['reviewed']*nP
                    pep_info['ProteinSequence'] = np.array(np.tile(new_protein_seq.iloc[0],(nP,1)))
                    e_peplist.loc[pep_find,e_peplist.columns[[10,gene_col]]] = np.array(np.tile(new_protein_ids.iloc[ia,:],(nP,1))) #Record which entries have been altered

        protein_names = pep_info.iloc[:,gene_col+int(not ProteinGrouping)*4]
        if len(np.unique(protein_names)) > 1:
            continue

        #print(pep_info)
        #print(e_peplist.iloc[q:(q-1+nP),0:RA])
        if othermains_bypeptide != '':
            #print(np.array(pep_effects).shape,final_pepmaineffectlist.iloc[q:q+nP,:].shape)
            final_pepmaineffectlist.iloc[q:q+nP,:] = np.array(pep_effects)
        final_peplist.iloc[q:q+nP, RA:] = np.array(pep_values)
        final_peplist.iloc[q:q+nP,0:RA] = np.array(pep_info.iloc[:,0:RA])
        #print(final_peplist.iloc[q:q-1+nP,:])
        print('#',q, i,pep_info.values[0,gene_col+int(not ProteinGrouping)*4])
        q = q + nP

    if regression_method == 'dataset':
        npeps2prots = protein_matrix.sum(axis=0)
        protein_matrix.drop(protein_matrix.columns[npeps2prots == 0],axis=1)
        protein_names = protein_matrix.columns
        q = final_peplist.shape[0]

    final_peplist = final_peplist.iloc[0:q-1,:]
    try:
        final_peplist = final_peplist.sort_values(by=['Charge'])
    except:
        final_peplist = final_peplist.sort_values(by=['Ions'])
    if othermains_bypeptide != '':
        final_pepmaineffectlist = final_pepmaineffectlist.iloc[0:q-1,:]
        final_pepmaineffectlist = final_pepmaineffectlist.sort_values(by=['Accession'])
    #missing_idx = np.isnan(final_peplist.iloc[:,RA:].astype(float))

    # Setup long-form table for fitting regression model
    ## Create vector of group names or continuous variable if ContGroup
    if ContGroup:
        Groups = np.tile(np.array(ContGroup),(1,(q-1))).flatten(order='F')
    else:
        Groups = np.tile(np.array(GroupNames),(1,(q-1))).flatten(order='F')

    ## Sort out new main effects, create long vectors of other variables
    if othermains_bypeptide != '':
        #e_maineffects_peptide = np.zeros(((q-1)*len(runIDs),nPmains))
        e_maineffects_peptide = np.tile(np.array(['aaaaaaaaaaaaaa']),((q-1)*len(runIDs),nPmains))
        for i in range(nPmains):
            e_maineffects_peptide[:,i] = np.tile(np.array(final_pepmaineffectlist.iloc[:,i]),(len(runIDs),1)).flatten(order='F')
        e_maineffects_peptide = pd.DataFrame(e_maineffects_peptide,columns = othermains_bypeptide_names)#.astype('category')
    if othermains_bysample != '':
        #e_maineffects_sample = np.zeros(((q-1)*len(runIDs),nSmains))
        e_maineffects_sample = np.tile(np.array(['aaaaaaaaaaaaaaa']),((q-1)*len(runIDs),nSmains))
        for i in range(nSmains):
            e_maineffects_sample[:,i] = np.tile(np.array(e_mainsampleeffects.iloc[:,i]),(1,(q-1))).flatten(order='F')
        e_maineffects_sample = pd.DataFrame(e_maineffects_sample,columns = othermains_bysample_names)#.astype('category')
    if othermains_bypeptide != '' and othermains_bysample != '':
        othermains = pd.concat([e_maineffects_sample,e_maineffects_peptide],axis=1,sort=False)
    elif othermains_bypeptide != '':
        othermains = e_maineffects_peptide
    elif othermains_bysample != '':
        othermains = e_maineffects_sample

    # Subject (or run) effects
    subject = np.tile(np.array(runIDs),(1,(q-1))).flatten(order='F')

    # Peptide effects
    #Peptides = np.tile(np.array(final_peplist.loc[:,final_peplist.columns[1]]),(len(runIDs),1)).flatten(order='F')
    Peptides = np.tile(np.array(final_peplist['PeptideID']),(len(runIDs),1)).flatten(order='F')

    # Get peptide and protein sequences for counting unique (by sequence) peptides for each protein and locating PTMs
    #PeptideSequence = np.tile(np.array(final_peplist.loc[:,final_peplist.columns[8]]),(len(runIDs),1)).flatten(order='F')
    PeptideSequence = np.tile(np.array(final_peplist['Sequence']),(len(runIDs),1)).flatten(order='F')
    #ProteinSequence = np.tile(np.array(final_peplist.loc[:,final_peplist.columns[5]]),(len(runIDs),1)).flatten(order='F')
    ProteinSequence = np.tile(np.array(final_peplist['ProteinSequence']),(len(runIDs),1)).flatten(order='F')

    # FDR-adjusted Mascot (or other metric of peptide identity confidence) scores
    #Scores = np.tile(np.array(final_peplist.loc[:,final_peplist.columns[7]].astype(float)/scorebf),(len(runIDs),1)).flatten(order='F')
    Scores = np.tile(np.array(final_peplist['Score'].astype(float)/scorebf),(len(runIDs),1)).flatten(order='F')
    Scores[Scores > 1] = 1

    Proteins = np.tile(np.array(final_peplist.iloc[:,gene_col+int(not ProteinGrouping)*4]),(len(runIDs),1)).flatten(order='F')

    Intensities = np.array(final_peplist.iloc[:,RA:].astype(float)).flatten(order='C')

    model_table = pd.DataFrame({'Protein':Proteins,
                                'Peptide':Peptides,
                                'PeptideSequence':PeptideSequence,
                                'Score':Scores,
                                'Group':Groups,
                                'Subject':subject,
                                'Intensities':Intensities,
                                'ProteinSequence':ProteinSequence},
                                index = list(range(q*len(runIDs)-len(runIDs))))

    # convert to categorical to save memory
    #model_table[['Protein','Peptide','PeptideSequence','ProteinSequence','Treatment','Subject']] =  model_table[['Protein','Peptide','PeptideSequence','ProteinSequence','Treatment','Subject']]#.astype('category')

    if othermains_bysample != '' or othermains_bypeptide != '':
        model_table = pd.concat([othermains,model_table],axis=1,sort = False)
    print(model_table)
    if form == 'maxquant':
        ContaminantsNReverses = pd.DataFrame({'Potential contaminant':np.tile(np.array(final_peplist['Potential contaminant']),(len(runIDs),1)).flatten(order='F'),
                                            'Reverse':np.tile(np.array(final_peplist['Reverse']),(len(runIDs),1)).flatten(order='F'),
                                            'Unique':np.tile(np.array(final_peplist['Unique (Proteins)']),(len(runIDs),1)).flatten(order='F')},
                                            index = model_table.index)
        model_table = pd.concat([model_table,ContaminantsNReverses],axis=1,sort = False)
        #print(model_table.columns.values)
        model_table = model_table.loc[(model_table['Potential contaminant'] != '+') & (model_table['Reverse'] != '+') & (model_table['Unique'] == 'yes')]
        #model_table = model_table.loc[,:]

    if regression_method == 'dataset':
        protein_matrix = pd.DataFrame(np.tile(protein_matrix,(nRuns,1)), columns = protein_names)
        model_table = pd.concat([protein_matrix,model_table],axis=1,sort = False)
        #print(model_table.columns)

    if impute == 'none':
        model_table = model_table.loc[model_table['Intensities'].notnull()]

    # Need to remove ProteinSequence for table or make it last column otherwise long protein sequences (e.g. TITIN) fuck up export to .tab and .csv
    return final_peplist.iloc[:,final_peplist.columns != 'ProteinSequence'], model_table, uniprotall, nGroups, nRuns

# Model fitting function; fits individual models for each protein
def fitProteinModels(model_table,otherinteractors,incSubject,subQuantadd,nGroups,nRuns,pepmin,random_effects,nChains,continuousvars=[]):

    #Make sure continuous variables are specified as floats
    model_table[continuousvars] = model_table[continuousvars].astype(np.float64)

    #Sort out group and subject variables
    unique_proteins = np.unique(model_table.loc[:,'Protein'])
    if 'Group' in continuousvars:
        t1 = ['Group']
        t = ['Group']
    else:
        t1 = list(np.unique(model_table.loc[:,'Group']))
        t2 = ['Group_']*len(t1)
        t = [m+str(n) for m,n in zip(t2,t1)]
    s = list(np.unique(model_table.loc[:,'Subject']))
    nProteins = len(unique_proteins)
    nInteractors = len(otherinteractors)

    #Preallocate dataframes to be filled by model fitting
    column_names = quantTableNameConstructor(t1,nRuns,isSubjectLevelQuant = False)
    ProteinQuant = pd.DataFrame(columns = ['Protein','# peptides','degrees of freedom','MSE']+column_names+['RhatMAX','% missing'])
    Othermains_column_types = model_table.columns[(model_table.columns != 'Treatment') * (model_table.columns != 'Peptide') * (model_table.columns != 'Protein') * (model_table.columns != 'PeptideSequence') * (model_table.columns != 'Score') * (model_table.columns != 'Subject') * (model_table.columns != 'ProteinSequence')]
    IsoformQuant = pd.DataFrame(columns = ['Parent Protein','Peptide','Scaled peptide score','degrees of freedom']+column_names)
    PTMQuant = pd.DataFrame(columns=['Peptide #','Parent protein','Peptide','Scaled peptide score','PTMed residue','PTM type','PTM position in peptide','PTM position in protein','degrees of freedom']+column_names)
    models = {} #Dictionary with all fitted protein models
    #PTMQuant = pd.concat([PTMQuant,pd.DataFrame(np.zeros((ntotalPTMs,nGroups*4)),columns = column_names)],axis=1,ignore_index=True,sort=False)

    # Summing user-specified effects for subject-level protein quantification
    SubjectLevelColumnNames = quantTableNameConstructor(s,nRuns,isSubjectLevelQuant = True)
    SubjectLevelProteinQuant = pd.DataFrame(columns = SubjectLevelColumnNames,index=[-1])
    SubjectLevelPTMQuant = pd.DataFrame(columns = SubjectLevelColumnNames,index=[-1])
    allValues = pd.DataFrame(columns = ['protein','peptide']+list(model_table['Subject'].iloc[0:nRuns]),index=[-1])
    missingValues = pd.DataFrame(columns = list(model_table['Subject'].iloc[0:nRuns]),index=[-1])

    #Fit protein models
    print('Fitting protein models...')
    q = 0
    w = 0
    #v = 0
    for protein in unique_proteins:
        '''
        if protein == 'PRI1_HUMAN':
            v = 1

        if v==0:
            continue
        '''
        start = time.time()
        protein_table = model_table.loc[model_table.loc[:,'Protein'] == protein,:].reset_index(drop=True)
        nPeptides = len(np.unique(protein_table['PeptideSequence']))
        nPeptides2 = len(np.unique(protein_table['Peptide']))

        if nPeptides < pepmin or protein == '':
            continue #Skip proteins with fewer than pepmin peptides in dataset or those composed of unassigned peptides
        #v = 1
        #Create design matrix (only add Treatment:Peptides interactions if more than 2 peptides)
        X = designMatrix(protein_table,otherinteractors,incSubject,len(np.unique(protein_table['Peptide'])))
        Y = np.array(protein_table.loc[:,'Intensities'])[:,np.newaxis]#.astype('float32')
        if np.all(np.isnan(Y)):
            continue
        parameterIDs = X.columns
        p = ['Peptide_'+str(n) for n in list(np.unique(protein_table.loc[:,'Peptide']))]

        #Identify user-specified random effects
        rand_eff_ids = []

        if random_effects == 'all':
            rand_eff_ids = X.columns
            fixed_eff_indices = []
        elif random_effects != [''] and random_effects != 'all':
            for effect in random_effects:
                rand_eff_ids = rand_eff_ids+[effect+'_'+str(n) for n in list(np.unique(protein_table.loc[:,effect]))]

        fixed_eff_indices = [X.columns.get_loc(column) for column in X if column not in rand_eff_ids]
        if nPeptides > 1:
            X_missing = X[t+p]#X[parameterIDs.intersection(t+p)]
        else:
            X_missing = X[t]

        missing = np.isnan(Y)
        Y_missing = (np.sign(missing.astype(float)-0.5)*10)#.astype('float32')
        parameterIDs_missing = X_missing.columns
        n = X_missing.shape[0]
        X_missing = np.array(X_missing,dtype=int)#sp.sparse.csc_matrix(np.array(X_missing,dtype=np.float32))
        X = np.array(X,dtype=int)#sp.sparse.csc_matrix(np.array(X,dtype=np.float32))
        percent_missing = [np.sum(missing)/n*100]

        # Fit missing model to choose which missing values are MNR and which are MAR
        missingmdl = weighted_bayeslm_multi(X_missing,Y_missing,parameterIDs_missing,False,np.ones([n,1]),np.array([]),nInteractors,[],nChains,200,0)
        #MNR = np.ravel(np.sum(np.concatenate((missingmdl['beta_estimate'] > 0, missingmdl['beta_estimate'] > missingmdl['b0']),axis=0),axis=0)) == 2
        MNR = np.ravel(np.sum(missingmdl['beta_estimate'] > 0,axis=0)) == 1
        Y_MNR = (X_missing[np.where(missing)[0],:][:,MNR].sum(axis=1) > 0)

        # Fit protein model to estimate fold changes while imputing missing values
        proteinmdl = weighted_bayeslm_multi(X,Y,parameterIDs,True,np.array(protein_table['Score'])[:,np.newaxis],Y_MNR,nInteractors,fixed_eff_indices,nChains,200,0)
        #b0SEM = list(proteinmdl['b0SEM'])
        results = proteinmdl['beta_estimate']
        SEMs = proteinmdl['SEMs']
        dof = proteinmdl['dof']
        Rhatmax = [np.nanmax(proteinmdl['Rhat'])]
        #Yimputed = Y
        #if sum(missing) > 0:
        Yimputed = proteinmdl['Yimputed'].reshape((-1,nRuns))
        #RhathatParam = parameterIDs[np.argmax(proteinmdl['Rhat'])]

        # Sort out protein-level effects
        Treatment_i = effectFinder(parameterIDs,'Group')
        Treatment_betas = list(results[Treatment_i])
        Treatment_SEMs = list(SEMs[Treatment_i])
        ProteinQuant = ProteinQuant.append(dict(zip(['Protein','# peptides','degrees of freedom','MSE']+column_names+['RhatMAX','% missing'],[protein,nPeptides,dof,proteinmdl['residVar']]+Treatment_betas+Treatment_SEMs+[1]*nGroups*2+Rhatmax+percent_missing)),ignore_index=True,sort=False) #We'll calculate p-values (Bayes Factors?) and FDR-adjusted p-values later on.

        peptideIDs = np.array(protein_table['Peptide']).reshape((Yimputed.shape[0],-1))[:,0].reshape((Yimputed.shape[0],1))
        allValuesRows = np.concatenate((np.array([protein]*Yimputed.shape[0]).reshape((Yimputed.shape[0],1)),peptideIDs,Yimputed),axis=1)
        allValues = allValues.append(pd.DataFrame(data=allValuesRows,columns=allValues.columns),ignore_index=True,sort=False)
        missingValues = missingValues.append(pd.DataFrame(data=missing.reshape((-1,nRuns)),columns=missingValues.columns),ignore_index=True,sort=False)

        # Sort out treatment:peptide interaction effects
        TreatmentPeptide_i = effectFinder(parameterIDs,'Group',True,'Peptide')
        TreatmentPeptide_names = parameterIDs[np.newaxis][TreatmentPeptide_i]
        TreatmentPeptide_betas = results[TreatmentPeptide_i]
        TreatmentPeptide_SEMs = SEMs[TreatmentPeptide_i]

        #Sort out other main effects
        OtherMains_names = np.array([])
        OtherMains_betas = np.array([])
        OtherMains_SEMs = np.array([])

        for main in Othermains_column_types:
            main_i = effectFinder(parameterIDs,main)
            main_names = parameterIDs[np.newaxis][main_i]
            main_betas = results[main_i]
            main_SEMs = SEMs[main_i]
            OtherMains_names = np.concatenate((OtherMains_names,main_names))
            OtherMains_betas = np.concatenate((OtherMains_betas,main_betas))
            OtherMains_SEMs = np.concatenate((OtherMains_SEMs,main_SEMs))

        nOtherMains = OtherMains_names.shape[0]
        OtherMains_columns = quantTableNameConstructor(list(OtherMains_names),nRuns)

        if w == 0:
            OtherMains_table = pd.DataFrame(columns = ['Protein','# peptides','degrees of freedom','MSE']+OtherMains_columns)
            w = 1

        OtherMainsrow = {'Protein':protein,
                  '# peptides':nPeptides,
                  'degrees of freedom':dof,
                  'MSE':proteinmdl['residVar'],
                  }
        OtherMainsrow.update(dict(zip(OtherMains_columns,list(OtherMains_betas)+list(OtherMains_SEMs)+[1]*nOtherMains*2)))
        OtherMains_table = OtherMains_table.append(OtherMainsrow,ignore_index=True,sort=False)

        ntotalPTMs = 0
        for peptide in np.unique(protein_table.loc[:,'Peptide']):

            PTMpositions_in_peptide = np.array(re.findall(r'\[([0-9]+)\]',peptide)).astype(int)-1 #PTM'd residues denoted by [#]
            peptide_finder = protein_table.loc[:,'Peptide'].isin([peptide])
            parent_protein_sequnce = protein_table.loc[peptide_finder,'ProteinSequence'].iloc[0]
            peptide_sequence = protein_table.loc[peptide_finder,'PeptideSequence'].iloc[0]
            peptide_position_in_protein = np.array([m.start() for m in re.finditer(peptide_sequence,parent_protein_sequnce)])

            '''
            try:
                peptide_position_in_protein = np.array([m.start() for m in re.finditer(peptide_sequence,parent_protein_sequnce)])
            except:
                continue
            '''
            if peptide_position_in_protein.size == 0:
                peptide_position_in_protein = 0

            peptide_score = np.mean(np.array(protein_table.loc[peptide_finder,'Score'])) #Uses mean Mascot score of all instances of that peptide, for MaxQuant use localisation score (FDR filter peptides beforehand and give unmodified peptides an arbitrarily high value?)?

            #Get effect value and SEM for peptide:treatment interaction
            beta_finder = list(filter(lambda x: x.endswith('Peptide_'+peptide), list(TreatmentPeptide_names))) #[beta.start() for beta in re.finditer('Peptide_'+peptide+'$',list(TreatmentPeptide_names))
            InteractionBetas = TreatmentPeptide_betas[[list(TreatmentPeptide_names).index(beta) for beta in beta_finder]] #[TreatmentPeptide_betas[i] for i in beta_finder]
            if InteractionBetas.size == 0:
                continue

            InteractionSEMs = TreatmentPeptide_SEMs[[list(TreatmentPeptide_names).index(beta) for beta in beta_finder]] #[TreatmentPeptide_SEMs[i] for i in beta_finder]
            Interactionvalues = list(InteractionBetas)+list(InteractionSEMs)+[1]*nGroups*2

            if len(PTMpositions_in_peptide) > 0:
                #Get rows with mod, modded residue, modded peptideID for each peptide
                PTMdResidues = [peptide[i] for i in PTMpositions_in_peptide]
                PTMs = re.findall(r'(?<=] )\S+',peptide)

                try:
                    PTMpositions_in_protein = PTMpositions_in_peptide+1 + peptide_position_in_protein
                    if peptide_position_in_protein == 0:
                        PTMpositions_in_protein = ['??']*len(PTMpositions_in_peptide)
                    else:
                        PTMpositions_in_protein = list(PTMpositions_in_protein)
                except:
                    PTMpositions_in_protein = ['Multiple']*len(PTMpositions_in_peptide)

                #Create new row for dataframe output table
                for residue in range(len(PTMpositions_in_peptide)):
                    #print(protein, peptide, PTMs,residue, len(PTMpositions_in_peptide))
                    PTMrow = {'Peptide #':ntotalPTMs,
                              'Parent protein':protein,
                              'Peptide':peptide+';',
                              'Scaled peptide score':peptide_score,
                              'PTMed residue':PTMdResidues[residue],
                              'PTM type':PTMs[residue],
                              'PTM position in peptide':PTMpositions_in_peptide[residue]+1,
                              'PTM position in protein':PTMpositions_in_protein[residue],
                              'degrees of freedom':dof}
                    PTMrow.update(dict(zip(column_names,Interactionvalues)))

                    PTMQuant = PTMQuant.append(PTMrow,ignore_index=True,sort=False)

                    if subQuantadd != [''] or incSubject:
                        # Subject-level PTM quantification by summing Treatment:Peptide interactions with user-specified:Peptide interaction terms
                        #subjectPTMQuant = np.concatenate((np.zeros((1,1)),results[TreatmentPeptide_i][np.newaxis]),axis=1).T
                        subjectTreatmentPTMQuant_i = effectFinder(parameterIDs,re.escape('Peptide_'+peptide)+r'(?![|])',True,re.escape('Treatment'))+effectFinder(parameterIDs,re.escape('Treatment'),True,re.escape('Peptide_'+peptide)+r'(?![|])')
                        subjectPTMQuant = results[subjectTreatmentPTMQuant_i][:,np.newaxis]
                        subjectPTMterms = np.zeros((1,nRuns))
                        if incSubject:
                            subjectPTMterms_i = effectFinder(parameterIDs,'Subject')
                            subjectPTMterms = results[subjectPTMterms_i]
                        subjectPTMterms = np.reshape(subjectPTMterms,(subjectPTMQuant.size,-1),order='F')

                        if subQuantadd != [''] and otherinteractors != {}:
                            for parameter in subQuantadd:
                                if parameter != 'Subject':
                                    subjectPTMQuant_i = effectFinder(parameterIDs,re.escape('Peptide_'+peptide)+r'(?![|])',True,re.escape(parameter))+effectFinder(parameterIDs,re.escape(parameter),True,re.escape('Peptide_'+peptide)+r'(?![|])')
                                    subjectPTMQuant_betas = np.tile(results[subjectPTMQuant_i][np.newaxis],(subjectPTMQuant.size,1))
                                    subjectPTMQuant = subjectPTMQuant + subjectPTMQuant_betas

                        subjectPTMQuant = subjectPTMQuant + subjectPTMterms
                        subjectPTMQuant = np.reshape(subjectPTMQuant,(-1,nRuns),order='F')
                        #subjectPTMQuant = np.ravel(subjectPTMQuant)

                        for ptm_num in range(subjectPTMQuant.shape[0]):#SubjectLevelPTMQuant = SubjectLevelPTMQuant.append(dict(zip(SubjectLevelColumnNames,subjectPTMQuant)),ignore_index=True,sort=False)
                            SubjectLevelPTMQuant.loc[SubjectLevelPTMQuant.index.max()+1,:] = subjectPTMQuant[ptm_num,:]

                ntotalPTMs = ntotalPTMs + len(PTMpositions_in_peptide)

            else:
                Isoformrow = {'Parent Protein':protein,
                              'Peptide':peptide,
                              'Scaled peptide score':peptide_score,
                              'degrees of freedom':dof}
                Isoformrow.update(dict(zip(column_names,Interactionvalues)))

                IsoformQuant = IsoformQuant.append(Isoformrow,ignore_index=True,sort=False)

        if subQuantadd != [''] or incSubject:
            #Sort out Subject-level protein quantification
            subjectLevelQuant = results[Treatment_i][np.newaxis].T
            subjectterms = np.zeros((1,nRuns))
            if incSubject:
                subjectterms_i = effectFinder(parameterIDs,'Subject')
                subjectterms = results[subjectterms_i]
            subjectterms = np.reshape(subjectterms,(subjectLevelQuant.size,-1),order='F')

            if subQuantadd != ['']:
                for parameter in subQuantadd:
                    if parameter != 'Subject':
                        subjectQuant_i = effectFinder(parameterIDs,parameter)
                        subjectQuant_betas = np.tile(results[subjectQuant_i][np.newaxis],(subjectLevelQuant.size,1))
                        subjectLevelQuant = subjectLevelQuant + subjectQuant_betas

            subjectLevelQuant = subjectLevelQuant + subjectterms
            subjectLevelQuant = np.reshape(subjectLevelQuant,(-1,nRuns),order='F')
            #SubjectLevelProteinQuant = SubjectLevelProteinQuant.append(dict(zip(SubjectLevelColumnNames,subjectLevelQuant)),ignore_index=True,sort=False)
            SubjectLevelProteinQuant.loc[SubjectLevelProteinQuant.index.max()+1,:] = subjectLevelQuant

        #Store model with all parameters
        models[protein] = proteinmdl
        timetaken = time.time()-start
        if ntotalPTMs > 0:
            print('#',q,'/',nProteins,protein,nPeptides,dof,proteinmdl['residVar'],Rhatmax,Treatment_betas,'+/-',Treatment_SEMs,'Found', ntotalPTMs,'PTM(s).', 'Took {:.2f} minutes.'.format(timetaken/60))# at ', [m+str(n) for m,n in zip(PTMdResidues,PTMpositions_in_protein)])
        else:
            print('#',q,'/',nProteins,protein,nPeptides,dof,proteinmdl['residVar'],Rhatmax,Treatment_betas,'+/-',Treatment_SEMs,'Found 0 PTM(s).', 'Took {:.2f} minutes.'.format(timetaken/60))
        q += 1

    #Clean up PTM quantification to account for different peptides (missed cleavages) that possess the same PTM at same site
    SubjectLevelPTMQuant = SubjectLevelPTMQuant.loc[0:,:]
    SubjectLevelProteinQuant = SubjectLevelProteinQuant.loc[0:,:]
    PTMQuant_cleaned,SubjectLevelPTMQuant_cleaned = cleanPTMquants(PTMQuant,nGroups,SubjectLevelPTMQuant)

    return ProteinQuant, PTMQuant_cleaned, IsoformQuant, SubjectLevelProteinQuant.astype(float), SubjectLevelPTMQuant_cleaned.astype(float), models, allValues, missingValues, OtherMains_table
'''
# Use PyMC3 to fit a single model for the entire dataset - very computationally intensive
def fitDatasetModel(model_table,otherinteractors,incSubject,subQuantadd,nGroups,nRuns,pepmin,random_effects):

    print('Fitting dataset model...')
    t1 = model_table.loc[:,'Treatment'].unique() #list(np.unique(model_table.loc[:,'Treatment']))
    t2 = ['Treatment_']*len(t1)
    t = [m+str(n) for m,n in zip(t2,t1)]
    nInteractors = len(otherinteractors)

    protein_list = np.array(model_table.loc[:,'Protein'].unique()) #np.unique(model_table.loc[:,'Protein'])
    nProteins = len(protein_list)

    X = designMatrix(model_table,otherinteractors,incSubject,len(model_table['Peptide'].unique()),regmethod='dataset')
    Y = np.array(model_table.loc[:,'Intensities'])[:,np.newaxis]
    parameterIDs = X.columns
    p = ['Peptide_'+str(n) for n in model_table['Peptide'].unique()]
    X_missing = X[parameterIDs.intersection(t+p)]
    missing = np.isnan(Y)
    Y_missing = (np.sign(missing.astype(float)-0.5)*10)
    parameterIDs_missing = X_missing.columns
    n = X_missing.shape[0]
    X_missing = sp.sparse.csc_matrix(np.array(X_missing,dtype=int))
    X = sp.sparse.csc_matrix(np.array(X,dtype=int))

    #Preallocate dataframes to be filled by model fitting
    column_names = quantTableNameConstructor(t1,nRuns,isSubjectLevelQuant = False)

    PTMQuant = pd.DataFrame(columns=['Peptide #','Parent protein','Peptide','Scaled peptide score','PTMed residue','PTM type','PTM position in peptide','PTM position in protein','degrees of freedom']+column_names)
    models = {} #Dictionary with all fitted protein models
    #PTMQuant = pd.concat([PTMQuant,pd.DataFrame(np.zeros((ntotalPTMs,nGroups*4)),columns = column_names)],axis=1,ignore_index=True,sort=False)

    # Summing user-specified effects for subject-level protein quantification
    SubjectLevelColumnNames = quantTableNameConstructor(t1,nRuns,isSubjectLevelQuant = True)
    SubjectLevelProteinQuant = pd.DataFrame(columns = SubjectLevelColumnNames)
    SubjectLevelPTMQuant = pd.DataFrame(columns = SubjectLevelColumnNames)

    # Fit missing model to choose which missing values are MNR and which are MAR
    missingtrace = pymc3_weightedbayeslm(X_missing,Y_missing,parameterIDs_missing,False,np.ones([n,1]),np.array([]),nInteractors)[1]
    missingb0 = np.array(pm.summary(missingtrace,varnames = ['b0'])['mean'])
    missingbeta = np.array(pm.summary(missingtrace,varnames = ['beta_estimate'])['mean'])
    MNR = np.ravel(np.sum(np.concatenate((missingbeta > 0, missingbeta > missingb0),axis=0),axis=0)) == 2
    Y_MNR = (X_missing[np.where(missing)[0],:][:,MNR].sum(axis=1) > 0)

    # Fit protein model to estimate fold changes while imputing missing values
    proteintrace = pymc3_weightedbayeslm(X,Y,parameterIDs,True,np.array(model_table['Score'])[:,np.newaxis],Y_MNR,nInteractors)[1]
    TreatmentPeptide_i = effectFinder(parameterIDs,'Treatment',True,'Peptide')
    ProteinTreatment_i = effectFinder(parameterIDs,'Protein_',True,'Treatment')
    betas = np.array(pm.summary(proteintrace,varnames = ['beta_estimate'])['mean'])
    SEMs = np.array(pm.summary(proteintrace,varnames = ['beta_estimate'])['sd'])
    MSE = np.tile(np.array(pm.summary(proteintrace,varnames = ['sigma2'])['mean']),(nProteins,1))

    # Sort out protein summary table
    DoF = np.tile(np.array(pm.summary(proteintrace,varnames = ['DoF'])['mean']),(nProteins,1))
    protein_betas = betas[ProteinTreatment_i].reshape((nProteins,-1))
    protein_SEMs = SEMs[ProteinTreatment_i].reshape((nProteins,-1))
    pvals = np.ones((nProteins,nGroups))
    protein_results = np.concatenate((protein_list,DoF,MSE,protein_betas,protein_SEMs,pvals),axis=1)
    ProteinQuant = pd.DataFrame(protein_results, columns = ['Protein','# unique peptides','degrees of freedom','MSE']+column_names)

    # Sort out PTM summary table
    TreatmentPeptide_betas = betas[TreatmentPeptide_i]
    TreatmentPeptide_SEMs = SEMs[TreatmentPeptide_i]
    TreatmentPeptide_names = parameterIDs[np.newaxis][TreatmentPeptide_i]

    ntotalPTMs = 0
    for peptide in model_table['Peptide'].unique():
        PTMpositions_in_peptide = np.array(re.findall(r'\[([0-9]+)\]',peptide)).astype(int)-1 #PTM'd residues denoted by [#]
        if len(PTMpositions_in_peptide) > 0:
            #Get rows with mod, modded residue, modded peptideID for each peptide
            PTMdResidues = [peptide[i] for i in PTMpositions_in_peptide]
            PTMs = re.findall(r'(?<=] )\S+',peptide)
            peptide_finder = model_table.loc[:,'Peptide'].isin([peptide])
            #parent_protein = model_table.loc[peptide_finder,'Protein'][0]
            parent_protein_sequnce = model_table.loc[peptide_finder,'ProteinSequence'].iloc[0]
            peptide_sequence = model_table.loc[peptide_finder,'PeptideSequence'].iloc[0]
            parent_protein = model_table.loc[peptide_finder,'Protein'].iloc[0]
            peptide_position_in_protein = np.array([m.start() for m in re.finditer(peptide_sequence,parent_protein_sequnce)])
            if peptide_position_in_protein.size == 0:
                peptide_position_in_protein = 0
            try:
                PTMpositions_in_protein = PTMpositions_in_peptide+1 + peptide_position_in_protein
                if peptide_position_in_protein == 0:
                    PTMpositions_in_protein = ['??']*len(PTMpositions_in_peptide)
                else:
                    PTMpositions_in_protein = list(PTMpositions_in_protein)
            except:
                PTMpositions_in_protein = ['Multiple']*len(PTMpositions_in_peptide)
            peptide_score = np.mean(np.array(model_table.loc[peptide_finder,'Score'])) #Uses mean Mascot score of all instances of that peptide, for MaxQuant use localisation score (FDR filter peptides beforehand and give unmodified peptides an arbitrarily high value?)?

            #Get effect value and SEM for modded peptide:treatment interaction
            #beta_finder = [list(TreatmentPeptide_names).index(beta) for beta in TreatmentPeptide_names if 'Peptide_'+peptide in beta]
            #pepmatch = re.compile(r''+peptide)
            #print(pepmatch,TreatmentPeptide_names,filter(pepmatch.search, list(TreatmentPeptide_names)))
            beta_finder = list(filter(lambda x: x.endswith(peptide), list(TreatmentPeptide_names))) #[beta.start() for beta in re.finditer('Peptide_'+peptide+'$',list(TreatmentPeptide_names))
            PTMbetas = TreatmentPeptide_betas[[list(TreatmentPeptide_names).index(beta) for beta in beta_finder]] #[TreatmentPeptide_betas[i] for i in beta_finder]
            #print(PTMbetas)
            if PTMbetas.size == 0:
                continue
            PTMSEMs = TreatmentPeptide_SEMs[[list(TreatmentPeptide_names).index(beta) for beta in beta_finder]] #[TreatmentPeptide_SEMs[i] for i in beta_finder]
            PTMvalues = list(PTMbetas)+list(PTMSEMs)+[1]*nGroups*2

            #Create new row for dataframe output table
            for residue in range(len(PTMpositions_in_peptide)):
                #print(protein, peptide, PTMs,residue, len(PTMpositions_in_peptide))
                PTMrow = {'Peptide #':ntotalPTMs,
                          'Parent protein':parent_protein,
                          'Peptide':peptide+';',
                          'Scaled peptide score':peptide_score,
                          'PTMed residue':PTMdResidues[residue],
                          'PTM type':PTMs[residue],
                          'PTM position in peptide':PTMpositions_in_peptide[residue]+1,
                          'PTM position in protein':PTMpositions_in_protein[residue],
                          'degrees of freedom':DoF}
                PTMrow.update(dict(zip(column_names,PTMvalues)))

                PTMQuant = PTMQuant.append(PTMrow,ignore_index=True,sort=False)

                if subQuantadd != ['']:
                    # Subject-level PTM quantification by summing Treatment:Peptide interactions with user-specified:Peptide interaction terms
                    #subjectPTMQuant = np.concatenate((np.zeros((1,1)),results[TreatmentPeptide_i][np.newaxis]),axis=1).T
                    subjectPTMQuant = betas[TreatmentPeptide_i][np.newaxis].T
                    for parameter in subQuantadd:
                        subjectPTMQuant_i = effectFinder(parameterIDs,re.escape('Peptide_'+peptide),True,re.escape(parameter))+effectFinder(parameterIDs,re.escape(parameter),True,re.escape('Peptide_'+peptide))
                        subjectPTMQuant_betas = np.tile(betas[subjectPTMQuant_i][np.newaxis],(subjectPTMQuant.size,1))
                        subjectPTMQuant = subjectPTMQuant + subjectPTMQuant_betas
                        subjectPTMQuant = np.reshape(subjectPTMQuant,(-1,1),order='F')
                        SubjectLevelPTMQuant = SubjectLevelPTMQuant.append(dict(zip(SubjectLevelColumnNames,subjectPTMQuant)),ignore_index=True,sort=False)

        ntotalPTMs = ntotalPTMs + len(PTMpositions_in_peptide)

    if subQuantadd != ['']:
        #Sort out Subject-level protein quantification
        subjectLevelQuant = betas[ProteinTreatment_i][np.newaxis].T
        for parameter in subQuantadd:
            subjectQuant_i = effectFinder(parameterIDs,parameter,True,'Protein')+effectFinder(parameterIDs,'Protein',True,parameter)
            subjectQuant_betas = np.tile(betas[subjectQuant_i][np.newaxis],(subjectLevelQuant.size,1))
            subjectLevelQuant = subjectLevelQuant + subjectQuant_betas
            subjectLevelQuant = np.reshape(subjectLevelQuant,(-1,1),order='F')
        subjectLevelQuant = np.reshape(subjectLevelQuant,(-1,nRuns),order='F')
        SubjectLevelProteinQuant = pd.DataFrame(subjectLevelQuant, columns = SubjectLevelColumnNames)

    PTMQuant_cleaned,SubjectLevelPTMQuant_cleaned = cleanPTMquants(PTMQuant,nGroups,SubjectLevelPTMQuant)

    return ProteinQuant,PTMQuant_cleaned,SubjectLevelProteinQuant.astype(float),SubjectLevelPTMQuant_cleaned.astype(float),models
'''
#Clean up PTM quantification to account for different peptides (missed cleavages) that possess the same PTM at same site
def cleanPTMquants(PTMQuant,nGroups,SubjectLevelPTMQuant = pd.DataFrame([])):

    PTMids = PTMQuant[['Parent protein','PTMed residue','PTM type','PTM position in protein']].astype(str).sum(axis=1)
    PTMQuant_cleaned = pd.DataFrame(columns = PTMQuant.columns)
    SubjectLevelPTMQuant_cleaned = pd.DataFrame(columns = SubjectLevelPTMQuant.columns)

    for i in PTMids.unique():
        ptm_finder = PTMids.isin([i])
        if np.sum(ptm_finder) > 1:
            ptms = PTMQuant.loc[ptm_finder,:]
            ptmssubject = SubjectLevelPTMQuant.loc[ptm_finder,:]
            PTMrow = ptms.mean(axis=0, numeric_only = True).T #collapse all summary values to averages
            PTMrow['Peptide'] = ptms['Peptide'].astype(str).sum() #list peptides used
            PTMrow['PTM position in peptide'] = list(ii for ii in ptms['PTM position in peptide'].astype(str)) #list peptides used
            PTMrow['Peptide #'] = list(ii for ii in ptms['Peptide #'].astype(str))
            PTMrow['Parent protein'] = ptms['Parent protein'].iloc[0]
            PTMrow['PTMed residue'] = ptms['PTMed residue'].iloc[0]
            PTMrow['PTM type'] = ptms['PTM type'].iloc[0]
            PTMrow['PTM position in protein'] = ptms['PTM position in protein'].iloc[0]
            PTMSubjectrow = ptmssubject.mean(axis=0, numeric_only = False) #collapse all subject values to averages
            PTMSEs = np.sqrt((ptms.iloc[:,9+nGroups:9+nGroups*2].values**2).sum(axis=0)) #collapse summary errors by square-root of sum of squared errors
            PTMrow[['{SE}' in ii for ii in PTMrow.index]] = PTMSEs
            #PTMrow[['{EB t-test p-value}' in ii for ii in PTMrow.index]] = 1
            #PTMrow[['{BHFDR}' in ii for ii in PTMrow.index]] = 1
            PTMQuant_cleaned = PTMQuant_cleaned.append(PTMrow, ignore_index = True, sort = False)
            SubjectLevelPTMQuant_cleaned = SubjectLevelPTMQuant_cleaned.append(PTMSubjectrow, ignore_index = True, sort = False)
        else:
            PTMrow = PTMQuant.loc[ptm_finder,:]
            #PTMrow[('{EB t-test p-value}' in ii for ii in PTMrow.columns)] = 1
            #PTMrow[('{BHFDR}' in ii for ii in PTMrow.columns)] = 1
            PTMSubjectrow = SubjectLevelPTMQuant.loc[ptm_finder,:]
            PTMQuant_cleaned = PTMQuant_cleaned.append(PTMrow, ignore_index = True, sort = False)
            SubjectLevelPTMQuant_cleaned = SubjectLevelPTMQuant_cleaned.append(PTMSubjectrow, ignore_index = True, sort = False)

    return PTMQuant_cleaned, SubjectLevelPTMQuant_cleaned

# Create design matrix for Treatment + Peptide + Treatment*Peptide + additional user-specified main and interaction effects.
def designMatrix(protein_table,interactors,incSubject,nPeptides,regmethod = 'protein'):

    # Create design matrix - all fixed effects... for now.
    if incSubject:
        if nPeptides > 1:
            X_table = protein_table.loc[:,protein_table.columns.isin(['Protein','ProteinSequence','PeptideSequence','Score','Intensities','Potential contaminant','Reverse','Unique'])!=True]
        else:
            X_table = protein_table.loc[:,protein_table.columns.isin(['Protein','ProteinSequence','Peptide','PeptideSequence','Score','Intensities','Potential contaminant','Reverse','Unique'])!=True]
    else:
        if nPeptides > 1:
            X_table = protein_table.loc[:,protein_table.columns.isin(['Protein','ProteinSequence','PeptideSequence','Score','Intensities','Subject','Potential contaminant','Reverse','Unique'])!=True]
        else:
            X_table = protein_table.loc[:,protein_table.columns.isin(['Protein','ProteinSequence','Peptide','PeptideSequence','Score','Intensities','Subject','Potential contaminant','Reverse','Unique'])!=True]

    if regmethod == 'dataset':
        protein_effects = effectFinder(X_table.columns, 'Protein_')
        X_main = pd.get_dummies(X_table.loc[:,(protein_effects != True).flatten()])
        X_main = pd.concat((X_table.loc[:,protein_effects.flatten()],X_main),axis=1,sort = False)
    else:
        X_main = pd.get_dummies(X_table)

    X_main_labs = X_main.columns
    #print(X_main_labs)
    q = X_main.shape[0]

    # Process interactions (specified as a dictionary e.g. {Interactor1:Interactor2,Interactor3:Interactor1})
    ## Default is alway Peptide:Treatment, others are user-specified
    X_interactors = np.zeros([q,1])
    X_interactors = pd.DataFrame(X_interactors)
    if interactors != 'none':
        for i in X_main_labs:
            if 'Group' in i and nPeptides > 2:
                for ii in X_main_labs:
                    if 'Peptide' in ii:
                        name = i+':'+ii
                        temp = pd.DataFrame({name:np.array(X_main[i])*np.array(X_main[ii])})
                        X_interactors = pd.concat([X_interactors,temp],axis=1,sort=False)

        for i in interactors:
            for ii in X_main_labs:
                if i in ii:
                    for iii in X_main_labs:
                        if interactors[i] in iii:
                            name = ii + ':' + iii
                            temp = pd.DataFrame({name:np.array(X_main[ii])*np.array(X_main[iii])})
                            X_interactors = pd.concat([X_interactors,temp],axis=1,sort=False)

    X_interactors = X_interactors.iloc[:,1:]
    DM = pd.concat([X_main,X_interactors],axis=1,sort=False)
    return DM

# Find effects of interest from fitted model beta_estimate using model parameterIDs
def effectFinder(effectIDs, pattern, findInteractors = False,interactor = ''):
    found = np.zeros((1,len(effectIDs)))#['']*len(effectIDs)

    if findInteractors:
        pattern = pattern+'.*:'+interactor+'.*'
    else:
        pattern = '(?<!:)'+pattern+'(?!.+:)'

    for m in range(len(effectIDs)):
        b = re.findall(pattern,effectIDs[m])
        if b:
            found[0,m] = 1#b

    return found == 1

# Formatting for final output dataframe column names
def quantTableNameConstructor(group_names,nRuns,isSubjectLevelQuant = False):

    log2FoldChanges = dc(group_names)
    SEs = dc(group_names)
    Ps = dc(group_names)
    BHFDRs = dc(group_names)

    if isSubjectLevelQuant:
        column_names = log2FoldChanges*int(nRuns/len(group_names))
        #for name in log2FoldChanges:
         #   column_names = column_names+[name]*int(nRuns/len(group_names))

        for i,j in zip(column_names,range(nRuns-1)):
            column_names[j] = i+str(j)

    else:
        for i,j in zip(group_names,range(len(group_names))):
            log2FoldChanges[j] = '{Log2('+i+') fold change}'
            SEs[j] = i+'{SE}'
            Ps[j] = i+'{EB t-test p-value}'
            BHFDRs[j] = i+'{BHFDR}'

        column_names = log2FoldChanges+SEs+Ps+BHFDRs

    return column_names

# Median  or quantile normalisation for peptide intensities to specified set of peptides
def normalise(X,Z=[],method='median'):
    if method == 'median':
        if not Z.empty:
            median_Z = np.nanmedian(Z,axis=0)
        else:
            median_Z = np.nanmedian(X,axis=0)
        normalised_X = X - median_Z
    elif method == 'quantile':
        if not Z.empty:
            Zdf = pd.DataFrame(data=Z)
            rank_mean = Zdf.stack().groupby(Zdf.rank(method='average',na_option='top').stack().astype(int)).mean()
        else:
            rank_mean = X.stack().groupby(X.rank(method='average',na_option='top').stack().astype(int)).mean()
        #rank_mean[np.isinf(rank_mean)] = 0
        normalised_X = X.rank(method='average').stack().astype(int).map(rank_mean).unstack()

    return normalised_X

# Generic Benjamini-Hochberg p-value corrector for vector of p-values p
def bhfdr(p):
    nanidx = np.isnan(p)
    p[nanidx] = 1
    m = np.sum(nanidx == False)
    p_ord = np.sort(p,axis=0)
    idx = np.argsort(p,axis=0).squeeze()
    fdr = dc(p)
    fdr_ord =  np.minimum(1,p_ord * (float(m)/np.array(list(range(1,m+1))+[np.nan]*np.sum(nanidx))))
    fdr_ord = np.minimum.accumulate(np.fliplr(fdr_ord[np.newaxis]))
    fdr_ord = np.fliplr(fdr_ord)
    fdr[idx] = fdr_ord
    return fdr

# Pull 'species' data from Uniprot where 'species' is a string with either 'human' or 'mouse' or a UniProt Proteome ID
def getUniprotdata(species, download = True):
    upcol = 11
    if download:
        if species == 'human':
            url = 'http://www.uniprot.org/uniprot/?sort=&desc=&compress=no&query=proteome:UP000005640&fil=&force=no&preview=true&format=tab&columns=id,entry%20name,protein%20names,genes,go,go(biological%20process),go(molecular%20function),go(cellular%20component),go-id,interactor,genes(PREFERRED),reviewed,sequence'
            upcol = 11
        elif species == 'mouse':
            url = 'http://www.uniprot.org/uniprot/?sort=&desc=&compress=no&query=proteome:UP000000589&fil=&force=no&preview=true&format=tab&columns=id,entry%20name,protein%20names,genes,go,go(biological%20process),go(molecular%20function),go(cellular%20component),go-id,interactor,genes(PREFERRED),database(MGI),reviewed,sequence'
            upcol = 12
        else:
            url = 'http://www.uniprot.org/uniprot/?sort=&desc=&compress=no&query=proteome:'+species+'&fil=&force=no&preview=true&format=tab&columns=id,entry%20name,protein%20names,genes,go,go(biological%20process),go(molecular%20function),go(cellular%20component),go-id,interactor,genes(PREFERRED),reviewed,sequence'
            upcol = 12

        request = urllib.request.Request(url)
        response = urllib.request.urlopen(request)
        uniprotall = response.read().decode('utf-8')

        with open("uniprotall.tab", "w") as tsv:
            print(uniprotall, file = tsv)

    updf = pd.read_table("uniprotall.tab")

    return updf, upcol

# Import PEAKS 10 protein-peptide.csv output from PEAKS label-free quant and protein-peptide.csv outputs from PEAKS DB, PEAKS PTM and SPIDER searches
# Before exporting from PEAKS, remember to set all filters (Score, -10lgP. etc) to zero to get all peptides, not just the ones PEAKS thinks are good.
def PEAKS2BENP(peplist):

    print('Importing',peplist,'peptide list: ')
    PEAKSdf = pd.read_csv(peplist[0])
    DBdf = pd.read_csv(peplist[1])
    PTMdf = pd.read_csv(peplist[2])
    SPIDERdf = pd.read_csv(peplist[3])

    #find where intensity columns belonging
    intensity_begin = int(np.where([i == 'Avg. Area' for i in PEAKSdf.columns])[0]+1)
    intensity_end = int(np.where([i == 'Sample Profile (Ratio)' for i in PEAKSdf.columns])[0])
    intensity_cols = PEAKSdf[PEAKSdf.columns[intensity_begin:intensity_end]] #replace 'intensity' with some other identifier for these columns

    #find column with peptide sequence, modifications, accession, charge, PEP, peptideID, etc
    Proteins = PEAKSdf[['Protein Group','Protein ID','Used','Candidate','Start','Significance','Avg. ppm','Quality','Peptide','PTM','Protein Accession','Avg. Area']]
    Proteins = Proteins.rename(columns={'Protein Group':'Reviewed?','Protein Accession': 'Accession', 'Used':'Charge', 'PTM':'Modifications','Significance':'ProteinSequence','Protein ID':'PeptideID','Peptide':'Sequence','Quality':'Score'})

    # Make progenesis-style peptide sequence and PTM summary for each peptide
    for i in range(Proteins.shape[0]):
        #if i < 22497:
        #    continue
        peptide_seq = Proteins['Sequence'].iloc[i]
        peptide_seq = re.sub(re.escape('(*)'),'',peptide_seq)
        peptide_seq = re.sub('[A-Z]{1,1}[\.](?=.{2,})','',peptide_seq,count=1) #peptide_seq[2:-2][\.][A-Z]
        peptide_seq = re.sub('(?=.{2,})[\.][A-Z]{1,1}','',peptide_seq,count=1)
        peptide_seq = peptide_seq
        escape_seq = re.escape(peptide_seq)
        reg_seq = '^'+escape_seq+'\..?|\.'+escape_seq+'\..?|\.'+escape_seq+'$'

        DBpepseq_finder = DBdf['Peptide'].str.contains(reg_seq)
        DBascore = DBdf['AScore'].loc[DBpepseq_finder]

        PTMpepseq_finder = PTMdf['Peptide'].str.contains(reg_seq)
        PTMascore = PTMdf['AScore'].loc[PTMpepseq_finder]

        SPIDERpepseq_finder = SPIDERdf['Peptide'].str.contains(reg_seq)
        SPIDERascore = SPIDERdf['AScore'].loc[SPIDERpepseq_finder]

        ascore = pd.concat((DBascore,PTMascore,SPIDERascore),axis=0,sort=False)
        split_seq = re.split('\(.+?\)',peptide_seq)

        try:
            split_ascore = ascore.iloc[0].split(';')
        except:
            try:
                split_ascore = ascore.iloc[0]
            except:
                split_ascore = ['']

        temp_seq = ''
        for j in range(len(split_seq)):
            temp_seq = temp_seq + split_seq[j]

        PTMsummary = ''
        mod_types = []
        mod_positions = []
        if split_seq != [peptide_seq]:
            for j in range(len(split_seq)-1):
                try:
                    mod_types = mod_types + [split_ascore[j].split(':')[1]]
                    mod_positions = mod_positions + re.findall('[0-9]+',split_ascore[j].split(':')[0])
                    PTMsummary = PTMsummary + ' [' + str(mod_positions[j]) + '] ' + mod_types[j]
                except:
                    continue

        Proteins['Sequence'].iloc[i] = temp_seq
        Proteins['Modifications'].iloc[i] = PTMsummary
        peptideID = temp_seq+PTMsummary
        Proteins['PeptideID'].iloc[i] = peptideID
        print(i,'/',Proteins.shape[0],peptideID)

    pepdf = pd.concat((Proteins,intensity_cols),axis=1,sort=False)
    GroupNames = list(intensity_cols.columns)
    runs = list(intensity_cols.columns)
    for i in range(len(GroupNames)):
        GroupNames[i] = re.sub(r'(.[0-9]+)','',GroupNames[i])
    #GroupNames = np.unique(GroupNames)
    print(GroupNames)

    return pepdf, GroupNames, runs, 12, PEAKSdf.shape[0], intensity_cols.shape[1]

# Import peptide tables from list 'peplist' consisting of a list of MaxQuant peptide .csv tables and associated modification tables
def MaxQuant2BENP(peplists):
    print('Importing',peplists,'peptide lists: ')
    MQdfs = [pd.read_table(peplist) for peplist in peplists]

    #Remove potential contaminants
    #MQdfs[0] = MQdfs[0].loc[MQdfs[0]['Potential contaminant'] != '+']

    #Get important values
    Intensities = MQdfs[0].loc[:,['Intensity ' in i for i in MQdfs[0].columns]]
    #print(Intensities.shape)
    #Convert to progenesis format
    try:
        Proteins = MQdfs[0][['Amino acid before','Protein group IDs','Charges','Potential contaminant','Reverse','PEP','First amino acid','Score','Sequence','Second amino acid','Leading razor protein','Unique (Proteins)']]
    except:
        Proteins = MQdfs[0][['Amino acid before','Protein group IDs','Charges','Potential contaminant','Reverse','PEP','First amino acid','Score','Sequence','Second amino acid','Leading razor protein','Unique (Proteins)']]
    Proteins = Proteins.rename(columns={'Amino acid before':'Reviewed?','Leading razor protein': 'Accession', 'Charges':'Charge', 'Second amino acid':'Modifications','Score':'ProteinSequence','Protein group IDs':'PeptideID', 'PEP':'Score'})

    #Get modID indices for searching in PTM tables
    #modIDs = MQdfs[0].iloc[:,-len(MQdfs):-1]
    if len(peplists) > 1:
        modIDs = MQdfs[0].loc[:,[' site IDs' in i for i in MQdfs[0].columns]]
        PTMdf = pd.DataFrame(['']*modIDs.shape[0],columns=['Modifications'])
        for i in range(modIDs.shape[0]):
            PTMsummary = '' #Progenesis-like PTM summary ([aa #] mod-type [aa #2] mod-type2)
            for j in range(modIDs.shape[1]):
                try:
                    ks = modIDs.iloc[i,j].split(';')
                except:
                    ks = [modIDs.iloc[i,j]]
                for k in ks:
                    if not np.isnan(float(k)):
                        modType = modIDs.columns[j][0:-9]
                        #print(int(np.where([l.find(modType+'Sites.txt')+1 for l in peplists])[0]))
                        modTypeList = MQdfs[int(np.where([l.find(modType+'Sites.txt')+1 for l in peplists])[0])]
                        #print('|',modType,'|', modTypeList.columns)
                        #modInPeptide = modTypeList['Modification window'][int(k)].split(';')#.replace(';','').find(modType)
                        modPosition = modTypeList['Position in peptide'][int(k)]
                        #for modPosition in range(len(modInPeptide)):
                        #   if modInPeptide[modPosition] is not 'X':
                        PTMsummary = PTMsummary + ' [' + str(modPosition) + '] ' + modType
            PTMdf.iloc[i] = PTMsummary

        Proteins['Modifications'] = PTMdf.reset_index(drop=True)

    pepdf = pd.concat((Proteins,Intensities),axis=1,sort=False)

    #Remove potential contaminants
    #pepdf = pepdf.loc[pepdf['Potential contaminant'] != '+']

    GroupNames = list(Intensities.columns)
    runs = list(Intensities.columns)
    for i in range(len(GroupNames)):
        GroupNames[i] = re.sub(r'_([0-9+])','',GroupNames[i])
    #GroupNames = np.unique(GroupNames)
    print(GroupNames)
    n = pepdf.shape[0]
    nRuns = len(runs)
    RAind = 12

    return pepdf, GroupNames, runs, RAind, n, nRuns

# Import peptide tables from peplist, a Progenesis QI formatted .csv file
def Progenesis2BENP(peplist):
    # import normalisation peptidelist
    print('Importing',peplist,'peptide list: ')
    with open(peplist, newline = '',errors='ignore') as csvfile:
        pepreader = csv.reader(csvfile)
        n = 0;

        for row in pepreader:
            #print(row)
            if 'Raw abundance' in row: #Find where intensities begin
                RAind = np.zeros((1,len(row)))
                #width = len(row)
                for i in range(len(row)):
                    RAind[0,i] = row[i].find("Raw abundance")+1
                RAind = (RAind == 1).nonzero()[1].flatten()[0]
                #print(RAind)
                #for i in RAind:
                #    if RAind[i] == 1:
            elif n == 1:  #Find group names
                GroupNames = row[RAind:]
                nRuns = len(GroupNames)
                name = ''
                for col in range(nRuns):
                    if GroupNames[col] == '':
                        GroupNames[col] = name;
                    else:
                        name = GroupNames[col]
                #print(GroupNames,nRuns)
            elif n == 2:  #Find column headers
                runs = row[RAind:]
            n = n + 1
            #print('=')

        print(n-2,' peptide (ion)s')
        pepdf = pd.read_csv(peplist,header = 2)
        pepdf = pepdf.rename(columns={'#':'Reviewed?','Retention time (min)':'PeptideID', 'm/z':'ProteinSequence'})
        return pepdf, GroupNames, runs, RAind, n-2,nRuns

# Weighted Bayesian regression function not typically called by user
# MNR must by boolean numpy array!!
def weighted_bayeslm(X,Y,featureIDs,do_weights,Scores,MNR,nInteractors,fixed_effect_ids, nIter, nBurn, seed, theta0=[]):

    [n,p] = X.shape
    meanY = np.nanmean(Y).flatten()
    stdY = np.nanstd(Y).flatten()
    sigma2_shape = (n-1+p)/2;
    beta_posterior = np.zeros((nIter-nBurn,p))
    DoF = np.zeros((nIter-nBurn,1))
    sigma_squared = np.zeros((nIter-nBurn,1))


    if theta0 == []:
        np.random.seed(int(seed*100000000))
        beta_estimate = 10*np.random.randn(1,p)
        sigma2 = np.random.gamma(sigma2_shape, 0.01)
        lambda_lasso = np.array(np.sqrt(np.random.gamma(p,1,(1,p))))
        lambda_ridge = np.array(np.random.gamma(1,1/3,(1,p)))
        tau_vector = np.random.rand(1,p)
        w = np.ones((n,1))
        wX = dc(X)
        wY = dc(Y)
        Yimputed = dc(Y)
    else:
        np.random.set_state(theta0['state'])
        beta_estimate = theta0['beta_posterior'][-1,:][np.newaxis]
        sigma2 = theta0['sigma2_posterior'][-1]
        lambda_lasso = theta0['lambda_lasso']
        lambda_ridge = theta0['lambda_ridge']
        tau_vector = theta0['tau_vector']
        w = theta0['weights']
        Yimputed = theta0['Yimputed']
        wY = Y*w
        wX = X*w

    D_tau_squared = np.diag(tau_vector.flatten())
    s = np.random.binomial(1,Scores)+0.01
    XtX = wX.T @ wX

    # Imputation variables
    MNRimpmin = (np.nanmin(wY)-2-meanY)/(stdY+1)
    Ymissing = np.isnan(wY)
    Ymissing_i = np.where(Ymissing)[0]
    nMissing = Ymissing_i.size
    nMNR = int(np.sum(MNR,0))
    nMR = nMissing - nMNR
    prop_MNR = nMNR/n
    impY = np.full((nMissing,nIter-nBurn),np.nan)
    random_effect_ids = [i for i in list(range(p)) if i not in fixed_effect_ids]

    if nMissing:
        alpha = np.percentile(wY[np.where(Ymissing == False)[0]],prop_MNR*100)
        MNRimpmax = (alpha-meanY)/(stdY+1)
        if MNRimpmax < -37:
            MNRimpmax = -37 # Avoid imputed -infs
        if nMR:
            Xmiss = X[Ymissing_i[np.where(MNR == False)[0]],:]   # MNR must be np.array
            XXTYMR = linalg.inv((Xmiss @ Xmiss.T)*np.eye(nMR))

    ii = 0
    for i in range(nIter+1):
        if nMissing:
            if nMNR:
                # Impute MNR missing values from truncated gaussian
                z = stats.truncnorm.rvs(MNRimpmin,MNRimpmax,loc=meanY,scale=stdY,size=(nMNR,1))
                wY[Ymissing_i[np.where(MNR)[0]]] = z*w[Ymissing_i[np.where(MNR)[0]]]
            if nMR:
                # Impute MR missing values from multivariate_normal
                B = sigma2*XXTYMR
                D = (Xmiss @ beta_estimate.T).flatten()
                wY[Ymissing_i[np.where(MNR == False)[0]]] = (w[Ymissing_i[np.where(MNR == False)[0]]].T*mvn(D,B)).T

            Yimputed[Ymissing] = wY[Ymissing]/w[Ymissing]

        # beta_estimate from conditional multivariate_normal
        L = linalg.inv(np.diag((lambda_ridge+tau_vector).ravel())+XtX)
        C = L*sigma2#+(np.eye(p)*0.00001)
        #if do_weights:
        #    print(i,np.linalg.eigvals(C)[164])
        #    print(np.where(C[164]<0))
        #if np.linalg.det(C) < 0:
        #    C = C + np.eye(p)*0.01
        A = np.ndarray.flatten(L @ (wX.T @ wY))
        beta_estimate[:,fixed_effect_ids] = A[fixed_effect_ids]#mvn(A[fixed_effect_ids],C[fixed_effect_ids][:,fixed_effect_ids]).ravel()#[np.newaxis]
        beta_estimate[:,random_effect_ids] = mvn(A[random_effect_ids],C[random_effect_ids][:,random_effect_ids]).ravel()

        # sigma**2 from inverse gamma
        residuals = Yimputed - ((X[:,random_effect_ids] @ beta_estimate[:,random_effect_ids].flatten()[:,np.newaxis]) + (X[:,fixed_effect_ids] @ beta_estimate[:,fixed_effect_ids].flatten()[:,np.newaxis])) #np.concatenate((X0,X),axis=1) @ np.concatenate((b0,beta_estimate),axis=1).T
        sigma2_scale = (residuals.T @ residuals)/2 + ((linalg.inv(D_tau_squared) @ (beta_estimate*lambda_lasso).T).T @ beta_estimate.T)/2 + ((beta_estimate*lambda_ridge) @ beta_estimate.T)/2
        sigma2 = 1/np.random.gamma(sigma2_shape,1/sigma2_scale+0.01)

        # 1/tau**2 from IG
        tau2_shape = np.sqrt((lambda_lasso**2*sigma2)/beta_estimate**2)
        tau2_scale = lambda_lasso**2
        tau_vector[0,:] = np.random.wald(tau2_shape,tau2_scale)
        D_tau_squared = np.diag(tau_vector.ravel())

        # lambda_lasso and lambda_ridge from gamma
        lambda_lasso[0,:] = np.sqrt(np.random.gamma(p+nInteractors, 1/((1/tau_vector).sum()/2)+1))
        lambda_ridge[0,:] = np.random.gamma(1+nInteractors, 1/(beta_estimate**2/2/sigma2+(1/p))+3)

        if i > nBurn:
            beta_posterior[ii,:] = beta_estimate
            DoF[ii] = np.sum((X @ L @ X.T).diagonal())
            sigma_squared[ii] = sigma2
            impY[:,ii] = Yimputed[Ymissing]
            ii = ii + 1

        if do_weights:
            r = 1/(0.5+residuals**2/2/sigma2)
            s = np.random.binomial(1,Scores)
            w = 1+s+np.random.gamma(0.5+s, r)
            wY = Yimputed*w
            wX = X*w
            XtX = wX.T @ wX

    impY = np.nanmean(impY,1)
    Yimputed[Ymissing] = impY

    beta_SEMs = np.std(beta_posterior,axis=0)
    beta_estimate = np.mean(beta_posterior,axis=0)
    #b0 = np.mean(intercept,axis=0)
    #b0SEM = np.std(intercept,axis=0)
    sigma2 = np.mean(sigma_squared)
    yfit = X @ beta_estimate.T
    #tscores = np.abs(beta_estimate)/SEMs
    #dof = beta_estimate[beta_estimate!=0].size
    dof = np.mean(DoF)
    #pvalues = (1 - sp.stats.t.cdf(tscores,dof)) * 2

    return {'beta_posterior':beta_posterior,
            #'b0_posterior':intercept,
            'beta_estimate':beta_estimate[np.newaxis],
            #'b0':b0,
            'tau_vector':tau_vector,
            'weights':w,
            'lambda_lasso':lambda_lasso,
            'lambda_ridge':lambda_ridge,
            'SEMs':beta_SEMs[np.newaxis],
            #'b0SEM':b0SEM,
            'Yimputed':Yimputed,
            'sigma2_posterior':sigma_squared,
            'residVar':sigma2,
            'yfit':yfit,
            'dof':dof,
            'parameters':featureIDs,
            'state':np.random.get_state()}#,

def weighted_bayeslm_multi(X,Y,featureIDs,do_weights,Scores,MNR,nInteractors,fixed_effect_ids,nChains,nIter = 1500, nBurn = 750):

    if nChains == 1:
        mdl = weighted_bayeslm(X,Y,featureIDs,do_weights,Scores,MNR,nInteractors,fixed_effect_ids,1500,750,1,[])
        return mdl

    Y_missing = np.isnan(Y)
    seeds = list(range(nChains))
    # Initial run
    pool = multiprocessing.Pool(nChains)
    #jobs = pool.apply_async(weighted_bayeslm, args=(X,Y,featureIDs,do_weights,Scores,MNR,nInteractors,fixed_effect_ids))
    #with multiprocessing.Pool(nChains) as pool:
    params = [[X,Y,featureIDs,do_weights,Scores,MNR,nInteractors,fixed_effect_ids,nIter,0]]*nChains
    params = [params[i]+[seeds[i]] for i in range(nChains)]
    try:
        jobs = pool.starmap(weighted_bayeslm, (params))# for i in range(nChains)]
    except:
        seeds = [s+nChains for s in seeds]
        params = [[X,Y,featureIDs,do_weights,Scores,MNR,nInteractors,fixed_effect_ids,nIter,nBurn]]*nChains
        params = [params[i]+[seeds[i]] for i in range(nChains)]
        jobs = pool.starmap(weighted_bayeslm, (params))

    pool.close()
    pool.join()

    try:

        beta_posterior = np.array([i['beta_posterior'] for i in jobs])
        betas = np.array([i['beta_estimate'] for i in jobs])
        beta_estimate = np.mean(betas,axis=0)

    except multiprocessing.TimeoutError:
            print("Aborting due to timeout")
            #pool.terminate()
            raise

    # Gelman-Rubin (1992) diagnostic for convergence
    Bovern = (1/(nChains-1))*np.sum((np.mean(beta_posterior,axis=1)-beta_estimate)**2,axis=0);
    L = np.sum(np.array([i['beta_posterior'] - i['beta_estimate'] for i in jobs])**2,axis=1)
    W = (1/(nChains*(beta_posterior.shape[1]-1)))*L
    sigma2plus = (((beta_posterior.shape[1]-1)/beta_posterior.shape[1])*W)+Bovern;
    Vhat = sigma2plus+Bovern/nChains;
    d = 2*Vhat/np.var(Vhat)
    Rhat = ((d + 3)/(d + 1))*Vhat/W;
    Rhatmax = np.max(Rhat)
    print('Iterations:',nIter,', RhatMAX =',Rhatmax)
    #nBurn = 0
    #nIter = 100

    keepSampling = True
    maxIter = 1000
    # Keep adding 100 iterations until convergence
    while Rhatmax >= 1.1 and keepSampling:

        #if __name__ == 'BayesENproteomics':
        pool = multiprocessing.Pool(nChains)
        #jobs = pool.apply_async(weighted_bayeslm, args=(X,Y,featureIDs,do_weights,Scores,MNR,nInteractors,fixed_effect_ids))
        #with multiprocessing.Pool(nChains) as pool:
        params = [[X,Y,featureIDs,do_weights,Scores,MNR,nInteractors,fixed_effect_ids,nIter,nBurn]]*nChains
        params = [params[i]+[seeds[i]]+[jobs[i]] for i in range(nChains)]
        try:
            jobs = pool.starmap(weighted_bayeslm, (params))# for i in range(nChains)]
        except:
            seeds = [s+nChains for s in seeds]
            params = [[X,Y,featureIDs,do_weights,Scores,MNR,nInteractors,fixed_effect_ids,nIter,nBurn]]*nChains
            params = [params[i]+[seeds[i]] for i in range(nChains)]
            jobs = pool.starmap(weighted_bayeslm, (params))

        pool.close()
        pool.join()

        try:
            beta_posterior = np.concatenate((beta_posterior,np.array([i['beta_posterior'] for i in jobs])),axis=1)
            betas = np.mean(beta_posterior,axis=1)#np.array([i['beta_estimate'] for i in jobs])
            beta_estimate = np.mean(betas,axis=0)[np.newaxis]

        except multiprocessing.TimeoutError:
            print("Aborting due to timeout")
            pool.terminate()
            raise

        # Gelman-Rubin (1992) diagnostic for convergence
        Bovern = (1/(nChains-1))*(np.mean(beta_posterior,axis=1)-beta_estimate)**2;
        L = np.sum(np.array([beta_posterior[i,:,:]-betas[i,:] for i in range(nChains)])**2,axis=1)#np.sum(np.array([i['beta_posterior'] - i['beta_estimate'] for i in jobs])**2,axis=1)#
        W = (1/(nChains*(beta_posterior.shape[1]-1)))*L
        sigma2plus = (((beta_posterior.shape[1]-1)/beta_posterior.shape[1])*W)+Bovern;
        Vhat = sigma2plus+Bovern/nChains;
        Rhat = Vhat/W;
        Rhatmax = np.max(Rhat)
        #nIter = abs(int((Rhatmax-np.min(Rhat))*1500))
        #theta0 = dc(jobs)

        print('Iterations:',beta_posterior.shape[1],', RhatMAX =',Rhatmax)
        if beta_posterior.shape[1] > maxIter:
            keepSampling = False
            #nIter = 200

    nBurn = int((beta_posterior.shape[1])/2)
    betas = np.mean(beta_posterior[:,nBurn:-1,:],axis=1)#np.array([i['beta_estimate'] for i in jobs])
    beta_estimate = np.mean(betas,axis=0)[np.newaxis]
    beta_SEMs = np.mean(np.std(beta_posterior[:,nBurn:-1,:],axis=1),axis=0)[np.newaxis] #np.nanstd(betas,axis=0)
    Yimputed = np.mean(np.array([i['Yimputed'] for i in jobs]),axis=0)
    dof = np.mean(np.array([i['dof'] for i in jobs]))
    sigma2 = np.mean(np.array([i['residVar'] for i in jobs]))
    return {#'beta_posterior':beta_posterior,
                #'b0_posterior':intercept,
                'beta_estimate':beta_estimate,
                #'b0':b0,
                'SEMs':beta_SEMs,
                #'b0SEM':b0SEM,
                'Yimputed':Yimputed,
                'residVar':sigma2,
                #'yfit':yfit,
                'Rhat':Rhat,
                'dof':dof,
                'parameters':featureIDs}
    '''
    wX = dc(X)
    [n,p] = wX.shape
    wY = dc(Y)
    Yimputed = dc(Y)
    meanY = np.nanmean(Y).flatten()
    np.random.seed(1345)
    iNumIter = 1000
    iBurn = 500

    sigma2_shape = (n-1+p)/2;
    beta_posterior = np.zeros((iNumIter-iBurn,p,nChains))
    intercept = np.zeros((iNumIter-iBurn,nChains))
    sigma_squared = np.zeros((iNumIter-iBurn,nChains))
    DoF = np.zeros((iNumIter-iBurn,nChains))

    # Imputation variables
    impmin = np.nanmin(wY)-2
    Ymissing = np.isnan(wY)
    Ymissing_i = np.where(Ymissing)[0]
    nMissing = Ymissing_i.size
    nMNR = int(np.sum(MNR,0))
    nMR = nMissing - nMNR
    prop_MNR = nMNR/n
    impY = np.full((nMissing,iNumIter-iBurn,nChains),np.nan)
    #D = np.tile(meanY,nMR)
    if nMissing:
        alpha = np.percentile(wY[np.where(Ymissing == False)[0]],prop_MNR*100)
        if nMR:
            Xmiss = X[Ymissing_i[np.where(MNR == False)[0]],:]   # MNR must be np.array
            #if nMR/n < 0.9:
            #try:
            XXTYMR = linalg.inv((Xmiss @ Xmiss.T)*np.eye(nMR))
                #np.linalg.cholesky(XXTYMR)
            #else:
            #except:
            #    nMR = False # If inv fails probably means there are too many missing values -> protein is low abundance -> use MNR imputation
            #    MNR = np.array([True for i in MNR])
            #    nMNR = nMissing

    for chain in range(nChains):
        ii = 0
        tau_vector = np.random.rand(1,p)
        D_tau_squared = np.diag(tau_vector.flatten())
        XtX = X.T @ X
        w = np.ones((n,1))
        beta_estimate = np.random.randn(1,p)
        b0 = np.random.randn(1,1)
        sigma2 = 1/np.random.gamma(sigma2_shape, 0.01)
        lambda_lasso = np.array(np.sqrt(np.random.gamma(p,1,(1,p))))
        lambda_ridge = np.array(np.random.gamma(1,1/3,(1,p)))

        for i in range(iNumIter+1):
            if nMissing:
                if nMNR:
                    # Impute MNR missing values from truncated gaussian
                    z = stats.truncnorm.rvs(impmin,alpha,loc=meanY,scale=sigma2,size=(nMNR,1))
                    wY[Ymissing_i[np.where(MNR)[0]]] = z*w[Ymissing_i[np.where(MNR)[0]]]
                if nMR:
                    # Impute MR missing values from multivariate_normal
                    B = sigma2*XXTYMR
                    y0 = sigma2*(np.random.randn(1,1)+np.nanmean(beta_estimate))
                    D = (y0 + Xmiss @ beta_estimate.T).flatten() #np.ndarray.flatten(np.concatenate((X0[Ymissing_i[np.where(MNR == False)[0]],:],Xmiss),axis=1) @ np.concatenate((b0,beta_estimate),axis=1).T.flatten())
                    wY[Ymissing_i[np.where(MNR == False)[0]]] = (w[Ymissing_i[np.where(MNR == False)[0]]].T*mvn(D,B)).T

                Yimputed[Ymissing] = wY[Ymissing]/w[Ymissing]

            # beta_estimate from conditional multivariate_normal
            #L = sp.sparse.csc_matrix(np.diag((lambda_ridge+tau_vector).ravel()))
            #L = sp.sparse.linalg.inv(XtX+L)
            L = linalg.inv(np.diag((lambda_ridge+tau_vector).ravel())+XtX)
            C = L*sigma2
            #C = L.multiply(sigma2)
            A = np.ndarray.flatten(L @ (wX.T @ wY))
            beta_estimate = mvn(A,C).ravel()#[np.newaxis]
            b0 = sigma2*(np.random.randn(1,1)+np.nanmean(Yimputed))

            # sigma**2 from inverse gamma
            residuals = Yimputed - (b0 + X @ beta_estimate[:,np.newaxis]) #np.concatenate((X0,X),axis=1) @ np.concatenate((b0,beta_estimate),axis=1).T
            sigma2_scale = (residuals.T @ residuals)/2 + ((linalg.inv(D_tau_squared) @ (beta_estimate*lambda_lasso).T).T @ beta_estimate.T)/2 + ((beta_estimate*lambda_ridge) @ beta_estimate.T)/2
            sigma2 = 1/np.random.gamma(sigma2_shape,1/sigma2_scale+0.01) #Change to .ravel() ??

            # 1/tau**2 from IG
            tau2_shape = np.sqrt((lambda_lasso**2*sigma2)/beta_estimate**2)
            tau2_scale = lambda_lasso**2
            tau_vector[0,:] = np.random.wald(tau2_shape,tau2_scale)
            D_tau_squared = np.diag(tau_vector.ravel())

            # lambda_lasso and lambda_ridge from gamma
            lambda_lasso[0,:] = np.sqrt(np.random.gamma(p+nInteractors, 1/((1/tau_vector).sum()/2)))
            lambda_ridge[0,:] = np.random.gamma(1+nInteractors, 1/(beta_estimate**2/2/sigma2+0.01)+3)

            if i > iBurn:
                beta_posterior[ii,:,chain] = beta_estimate
                DoF[ii,chain] = np.sum((X @ L @ X.T).diagonal())
                intercept[ii,chain] = b0
                sigma_squared[ii,chain] = sigma2
                impY[:,ii,chain] = Yimputed[Ymissing]
                ii = ii + 1

            if do_weights:
                r = 1/(0.5+residuals**2/2/sigma2)
                s = np.random.binomial(1,Scores)
                w = 1+s+np.random.gamma(s+0.5, r)
                wY = Yimputed*w
                wX = X*w
                XtX = wX.T @ wX

    impY = np.nanmean(np.nanmean(impY,2),1)
    Yimputed[Ymissing] = impY

    beta_estimate = np.mean(beta_posterior,axis=0) #chain means
    #beta_SEMs = np.mean(np.std(beta_posterior,axis=0),axis=1)
    beta_SEMs = np.sqrt(np.sum(np.std(beta_posterior,axis=0)**2,axis=1))
    beta_estimate = np.mean(beta_estimate,axis=1) #overall means
    b0 = np.mean(intercept)
    b0SEM = np.std(intercept)
    sigma2 = np.mean(sigma_squared)
    yfit = X @ beta_estimate.T
    #tscores = np.abs(beta_estimate)/SEMs
    #dof = beta_estimate[beta_estimate!=0].size
    dof = np.mean(DoF)
    #pvalues = (1 - sp.stats.t.cdf(tscores,dof)) * 2
    '''

'''
def pymc3_weightedbayeslm(X,Y,featureIDs,do_weights,Scores,MNR,nInteractors):

    n,p = X.shape
    lambda_lasso = np.random.rand(1,p)
    lambda_ridge = np.random.rand(1,p)
    sigma2 = np.random.gamma(1,0.01)
    tau_vector = np.random.rand(1,p)
    #do_weights = True

    XtX = X.T @ X
    wX = dc(X)
    wY = dc(Y)
    Yimputed = dc(Y)
    w = np.ones((n,1))
    b0 = np.random.randn(1,1)
    beta_estimate = np.random.randn(1,p)

    bayes_lm = pm.Model()
    impmin = np.nanmin(wY)-2
    Ymissing = np.isnan(wY)
    Ymissing_i = np.where(Ymissing)[0]
    nMissing = Ymissing_i.size
    #print(nMissing, Ymissing[np.where(MNR == 0)[0]])
    nMNR = int(np.sum(MNR,0))
    nMR = nMissing - nMNR
    #print(nMR)
    prop_MNR = nMNR/n
    #print(nMissing,nMR)
    #impY = np.full((nMissing,iNumIter-iBurn),np.nan)
    if nMissing:
        alpha = np.percentile(wY[np.where(Ymissing == False)[0]],prop_MNR*100)
        if nMR:
            Xmiss = X[Ymissing_i[np.where(MNR == False)[0]],:]
            XXTYMR = matrix_inverse(T.dot(Xmiss,Xmiss.T)) # MNR must be np.array

    with bayes_lm:
        if nMissing:
            if nMNR:
                # Impute MNR missing values from truncated gaussian
                BoundedNormal = pm.Bound(pm.Normal, lower=impmin, upper=alpha)
                z = BoundedNormal('z', mu=np.nanmean(Y), sd=sigma2)
                wY[Ymissing_i[np.where(MNR)[0]]] = z*w[Ymissing_i[np.where(MNR)[0]]]
                #print(impmin/sigma2,alpha/sigma2,plo,phi,i,z,sigma2)
            if nMR:
                # Impute MR missing values from Normal
                if nMR > 1:
                    B = T.diag(sigma2*(XXTYMR+0.00001)*T.eye(nMR))
                else:
                    B = sigma2*(XXTYMR+0.00001)*T.eye(nMR)
                #weighted_MR_vals = pm.MvNormal('weighted_MR_vals', mu=b0+T.dot(Xmiss,beta_estimate.T), cov=B,shape=(nMR,1))
                wY[Ymissing_i[np.where(MNR == False)[0]]] = np.random.rand(nMR,1)*((b0 + Xmiss @ beta_estimate.T)*B).eval()

            #print(wY[Ymissing].shape,(1/w[Ymissing]).shape)
            Yimputed[Ymissing] = wY[Ymissing]/w[Ymissing]

        b0 = pm.Normal('b0', mu=np.mean(Yimputed), sd=sigma2,shape=(1,1))
        L = matrix_inverse(T.eye(p)*(lambda_ridge+tau_vector)+XtX)
        A = T.dot(L,T.dot(wX.T,wY))+0.0001
        C = L * sigma2
        beta_estimate = pm.MvNormal('beta_estimate', mu=A.flatten(), cov=C, shape = (1,p))
        residuals = Yimputed - T.dot(X,beta_estimate.T)-b0
        sigma2_scale = T.dot(residuals.T,residuals)/2 + T.dot(T.dot(matrix_inverse(T.eye(p)*tau_vector),(beta_estimate*T.sqrt(lambda_lasso)).T).T, beta_estimate.T)/2 + T.dot((beta_estimate*lambda_ridge), beta_estimate.T)/2
        sigma2 = pm.InverseGamma('sigma2',alpha = (n+p-1)/2,beta = 1/sigma2_scale+0.01, shape=(1,1))
        #print(C.eval(),A.eval())
        tau_vector = pm.Wald('tau_vector',mu =T.sqrt(lambda_lasso/beta_estimate**2*sigma2),lam=lambda_lasso,shape = (1,p))
        lambda_lasso = pm.Gamma('lambda_lasso',alpha = p,beta = 1+T.sum(1/tau_vector)/2,shape=(1,1))
        lambda_ridge = pm.Gamma('lambda_ridge',alpha = 1+nInteractors, beta = 1/(beta_estimate**2/2/sigma2+0.001)+0.1,shape=(1,p))
        DoF = pm.Constant('DoF',np.sum(T.dot(T.dot(X,L),X.T).diagonal()))

        if do_weights:
            r = 1/(0.1+residuals**2/2/sigma2)+0.00001
            #s = np.random.binomial(1,Scores)#
            s = pm.Multinomial('s',n=1,p=np.array([Scores,1-Scores]).squeeze().T,shape=(n,1))
            w = 1+s+pm.Gamma('w',alpha=0.5+s, beta=r,shape=(n,1))
            wY = Yimputed*w
            wX = X*w
            XtX = T.dot(wX.T,wX)

        Y_obs = pm.MvNormal('Y_obs', mu=b0+T.dot(X,beta_estimate.T), cov=sigma2, observed=Yimputed,shape=(n,1))
        trace = pm.sample(500,njobs=1,nchains=2,nuts_kwargs={'max_treedepth':20,'target_accept':0.6,'integrator':'three-stage'},n_init=200,init='advi+adapt_diag')

    return bayes_lm,trace
'''
def mvn(mean, cov):
    L = linalg.cholesky(cov,check_finite=False)
    Z = np.random.normal(size = mean.shape)

    return Z.dot(L) + mean

def JumpMethod(X):
    n,p = X.shape
    Y = n/2

    D = np.zeros((n+1,))
    J = np.zeros((n,))
    d = np.zeros((n,))
    Vi = linalg.inv((X.T @ X)*np.eye(p))
    sih_scores = np.zeros((n,))

    for k in range(1,n):
        clusterer = KMeans(n_clusters=k, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        cluster_centres = clusterer.cluster_centers_
        cluster_centres_mat = np.array([cluster_centres[i] for i in cluster_labels])
        if k > 1:
            sih_scores[k] = silhouette_score(X, cluster_labels)

        for dim in range(n):
            #Vi = linalg.inv((X[:,dim][np.newaxis].T @ X[:,dim][np.newaxis])*np.eye(n))
            #Vi = np.eye(n)
            d[dim] = distance.mahalanobis(X[dim,:], cluster_centres_mat[dim,:], Vi)**(-Y)
        D[k] = np.mean(d)
        J[k] = D[k] - D[k-1]

    fig = mpl.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    ax.plot(range(1,n),J[1:], label = 'Delta D')
    ax.plot(range(1,n),D[2:], label = 'Transformed Distortion')
    ax.legend(loc='best', shadow=False, scatterpoints=1)
    ax.set_ylabel('Value')
    ax.set_xlabel('K')

    ax2 = fig.add_subplot(122)
    ax2.plot(range(1,n),sih_scores[1:])
    ax2.set_ylabel('Average Silhouette Score')
    ax2.set_xlabel('K')
    print(np.nanargmax(J),np.nanargmax(sih_scores))

    fig.savefig('JumpMethodFigures.pdf')
    return np.nanargmax(J)

def Impute(X,method,RA):
    intensities = np.array(X.iloc[:,RA:])

    # Down-shift Gaussian distribution (MNR)
    if method == 'dgd':
        print('Using imputation from Down-shifted Gaussian Distribution')
        means = np.nanmean(intensities,axis=1)
        sigmas = np.nanstd(intensities,axis=1)
        mus = means - (1.6*sigmas)

        for i in range(X.shape[0]):
            iMissing = np.isnan(intensities[i,:])
            nMissing = np.sum(iMissing)
            r = np.random.randn(1,nMissing)
            intensities[i,iMissing] = (0.3 * sigmas[i]) * r + mus[i]
            #print(intensities[i,iMissing])

    X.iloc[:,RA:] = intensities

    return X

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg
