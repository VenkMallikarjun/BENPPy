import csv
import numpy as np
import scipy as sp
from copy import deepcopy as dc
import urllib.request
import pandas as pd
import re
import matplotlib.pyplot as mpl
import pymc3 as pm
import theano.tensor as T
from theano.tensor.nlinalg import matrix_inverse
import os
#from theano import sparse
#import multiprocessing
import time
#from astropy.table import Table, Column

# Output object that hold all results variables
class BayesENproteomics:
    
    # Create empty object to be filled anew with doAnalysis() or from a saved folder by load()
    def __init__(self, output_name='output'):
        self.output_name = output_name
        if not os.path.exists(output_name):
            os.makedirs(output_name)
    
    # Wrapper for model fitting
    def doAnalysis(self, normalisation_peptides, experimental_peptides, organism, othermains_bysample = '',othermains_bypeptide = '', otherinteractors = {}, regression_method = 'protein', normalisation_method='median', pepmin=3, ProteinGrouping=False, peptide_BHFDR=0.2, nDB=1, incSubject=False, subQuantadd = [''], ContGroup=[]):
        
        if regression_method == 'dataset':
            otherinteractors['Protein_'] = 'Treatment'
            bayeslm = fitDatasetModel
        elif regression_method== 'protein':
            bayeslm = fitProteinModels
        
        peptides_used, longtable, missing_peptides_idx, UniProt, nGroups, nRuns = formatData(normalisation_peptides, experimental_peptides, organism, othermains_bysample, othermains_bypeptide, normalisation_method, ProteinGrouping, peptide_BHFDR, nDB, ContGroup)
        self.peptides_used = peptides_used
        self.missing_peptides_idx = missing_peptides_idx
        self.UniProt = UniProt
        self.longtable = longtable
        
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
        
        self.peptides_used.to_csv(self.output_name+'\\peptides_used.csv', encoding='utf-8', index=False,na_rep='nan',header=peptides_used.columns)
        self.input_table.to_csv(self.output_name+'\\input_table.csv', encoding='utf-8', index=False,header=input_table.columns)
        self.missing_peptides_idx.to_csv(self.output_name+'\\missing_peptides_idx.csv', encoding='utf-8', index=False,header=missing_peptides_idx.columns)
        self.UniProt.to_csv(self.output_name+'\\UniProt.csv', encoding='utf-8', index=False,header=UniProt.columns)
        self.longtable.to_csv(self.output_name+'\\longtable.csv', encoding='utf-8', index=False,header=longtable.columns)
        
        # Protein qunatification: fit protein or dataset model
        protein_summary_quant, PTM_summary_quant, protein_subject_quant, PTM_subject_quant, models = bayeslm(longtable,otherinteractors,incSubject,subQuantadd,nGroups,nRuns,pepmin)
        protein_list = list(protein_summary_quant.iloc[:,0].values.flatten())
        protein_info = pd.DataFrame(columns = UniProt.columns[[0,2,-2]])
        
        # Append protein information used in pathway analysis
        for protein in protein_list:
            protein_info_idx = self.UniProt.iloc[:,1].isin([protein])
            if np.any(protein_info_idx):
                protein_ids = self.UniProt.loc[protein_info_idx,self.UniProt.columns[[0,2,-2]]]
                protein_info = protein_info.append(protein_ids.iloc[0,:])
        self.protein_summary_quant = pd.concat((protein_summary_quant,protein_info.reset_index(drop=True)),axis=1,sort=False)

        protein_list = list(PTM_summary_quant.iloc[:,1].values.flatten())
        protein_info = pd.DataFrame(columns = UniProt.columns[[0,2,-2]])
        for protein in protein_list:
            protein_info_idx = self.UniProt.iloc[:,1].isin([protein])
            if np.any(protein_info_idx):
                protein_ids = self.UniProt.loc[protein_info_idx,self.UniProt.columns[[0,2,-2]]]
                protein_info = protein_info.append(protein_ids.iloc[0,:])
        self.PTM_summary_quant = pd.concat((PTM_summary_quant,protein_info.reset_index(drop=True)),axis=1,sort=False)

        # Empirical Bayes variance correction
        self.protein_summary_quant = EBvar(self.protein_summary_quant)[0]
        self.PTM_summary_quant = EBvar(self.PTM_summary_quant)[0]
        self.protein_subject_quant = protein_subject_quant
        self.PTM_subject_quant = PTM_subject_quant
        self.protein_summary_quant.to_csv(self.output_name+'\\protein_summary_quant.csv', encoding='utf-8', index=False,header=self.protein_summary_quant.columns)
        self.protein_subject_quant.to_csv(self.output_name+'\\protein_subject_quant.csv', encoding='utf-8', index=False,header=self.protein_subject_quant.columns)
        self.PTM_summary_quant.to_csv(self.output_name+'\\PTM_summary_quant.csv', encoding='utf-8', index=False,header=self.PTM_summary_quant.columns)
        self.PTM_subject_quant.to_csv(self.output_name+'\\PTM_subject_quant.csv', encoding='utf-8', index=False,header=self.PTM_subject_quant.columns)
        
        # Pathway quantification: fit pathway models
        pathway_summary_quant, pathway_models, Reactome = fitPathwayModels(self.protein_summary_quant, self.UniProt, organism, longtable, nRuns, False, self.protein_subject_quant)
        pathway_summary_quant = EBvar(pathway_summary_quant)[0]
        self.pathway_summary_quant = pathway_summary_quant
        self.Reactome = Reactome     
        
        self.pathway_summary_quant.to_csv(self.output_name+'\\pathway_summary_quant.csv', encoding='utf-8', index=False,header=self.pathway_summary_quant.columns)
        self.Reactome.to_csv(self.output_name+'\\Reactome.csv', encoding='utf-8', index=False,header=self.Reactome.columns)
    
    # Load exproted BayeENproteomics object from 'output_name' folder made during its creation
    def load(self):
        UniProt = pd.read_table(self.output_name+'\\UniProt.csv',sep = ',')
        peptides_used = pd.read_table(self.output_name+'\\peptides_used.csv',sep = ',')
        missing_peptides_idx = pd.read_table(self.output_name+'\\missing_peptides_idx.csv',sep = ',')
        input_table = pd.read_table(self.output_name+'\\input_table.csv',sep = ',')
        longtable = pd.read_table(self.output_name+'\\longtable.csv',sep = ',')
        self.UniProt = UniProt
        self.peptides_used = peptides_used
        self.missing_peptides_idx = missing_peptides_idx
        self.input_table = input_table
        self.longtable = longtable
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
            Reactome = pd.read_table(self.output_name+'\\Reactome.csv',sep = ',')
            self.Reactome = Reactome
        except:
            print('Reactome pathway data missing')
        
    # Compare all proteins between two experimental groups, repeated calls overwrite previous self.Constrasted 
    def doContrasts(self, Contrasted = 'protein', ctrl=0, propagateErrors=False,UseBayesFactors=False):
        
        if Contrasted == 'protein':
            self.Contrasted = dc(self.protein_summary_quant)
        elif Contrasted == 'ptm':
            self.Contrasted = dc(self.PTM_summary_quant)
        elif Contrasted == 'pathway':
            self.Contrasted = dc(self.pathway_summary_quant)
        else:
            msg = 'Contrasted must be ''pathway'', ''protein'' or ''ptm''.'
            raise InputError(Contrasted,msg)
            
        DoFs = np.array(self.Contrasted['degrees of freedom'])[:,np.newaxis]
        FCcolumns = np.array(self.Contrasted.iloc[:,['fold change}' in i for i in self.Contrasted.columns]])
        FCs = FCcolumns - FCcolumns[:,ctrl][:,np.newaxis]
        nProteins, nGroups = FCcolumns.shape
        SEs = np.array(self.Contrasted.iloc[:,['{SE}' in i for i in self.Contrasted.columns]])
        if propagateErrors:
            SEs[:,ctrl+1:] = np.sqrt(SEs[:,ctrl+1:]**2 + SEs[:,ctrl][:,np.newaxis]**2)
            if ctrl:
                SEs[:,0:ctrl] = np.sqrt(SEs[:,0:ctrl]**2 + SEs[:,ctrl][:,np.newaxis]**2)
                
        if UseBayesFactors:
            BF = FCs**2/(2*SEs**2)
            self.Contrasted.iloc[:,['{EB t-test p-value}' in i for i in self.Contrasted.columns]] = BF
        else:
            t = abs(FCs)/SEs
            pvals = np.minimum(1,sp.stats.t.sf(t, DoFs - 1) * 2 + 1e-15)
            fdradjp = dc(pvals)
        
            for i in range(nGroups):
                fdradjp[:,i] = bhfdr(pvals[:,i])
            
            self.Contrasted.iloc[:,['{EB t-test p-value}' in i for i in self.Contrasted.columns]] = pvals
            self.Contrasted.iloc[:,['{BHFDR}' in i for i in self.Contrasted.columns]] = fdradjp
            
        self.Contrasted.iloc[:,['fold change}' in i for i in self.Contrasted.columns]] = FCs
        self.Contrasted.iloc[:,['{SE}' in i for i in self.Contrasted.columns]] = SEs
            

    #PLOTS PLOTS PLOTS PLO-PLOTS PLOTS
    def boxplots(self): 
        nG = np.array(self.input_table['group number'])[0]
        fig1, ax1 = mpl.subplots()
        mpl.boxplot(self.protein_summary_quant.iloc[:,4:4+nG].T,notch=True,labels=self.protein_summary_quant.columns[4:4+nG])
        ax1.set_title('Protein fold changes')
        mpl.ylabel(r'$Log_2 (condition / ctrl)$')
        fig2, ax2 = mpl.subplots()
        mpl.boxplot(self.PTM_summary_quant.iloc[:,9:9+nG].T,notch=True,labels=self.PTM_summary_quant.columns[9:9+nG])
        ax2.set_title('PTM site occupancy fold changes')
        mpl.ylabel(r'$Log_2 (condition / ctrl)$')
        fig3, ax3 = mpl.subplots()
        mpl.boxplot(self.pathway_summary_quant.iloc[:,5:5+nG].T,notch=True,labels=self.pathway_summary_quant.columns[5:5+nG])
        ax3.set_title('Pathway effect size')
        mpl.ylabel(r'$Log_2 (condition / ctrl)$')

    
# Wrapper for pathway model fitting
def fitPathwayModels(models,uniprotall,species,model_table,nRuns,isPTMfile=False,subjectQuant=[]):
    
    print('Getting Reactome Pathway data...')
    url = 'http://reactome.org/download/current/UniProt2Reactome.txt'
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    uniprot2reactome = response.read().decode('utf-8')
    
    if species == 'human':
        species = 'Homo sapiens'
    elif species == 'mouse':
        species = 'Mus musculus'
    else:
        url = 'https://www.uniprot.org/proteomes/'+species
        request = urllib.request.Request(url)
        response = urllib.request.urlopen(request)
        species = response.read().decode('utf-8')
        species = re.search('<title>(\w+ \w+) (',species)
        
    with open("UniProt2Reactome.txt", "w",encoding="utf-8") as txt:
        print(uniprot2reactome, file = txt)

    uniprot2reactomedf = pd.read_table("UniProt2Reactome.txt",header=None)
    uniprot2reactomedf = uniprot2reactomedf.loc[uniprot2reactomedf.iloc[:,-1].isin([species]),:].reset_index(drop=True)
    print('Got Reactome pathway data')
    
    # Annotate pathway file with Identifiers used in protein quant
    u2r_protein_id = pd.DataFrame({'Entry name':['']*uniprot2reactomedf.shape[0]}) #columns = ['Entry name'])
    for uniprotID in list(uniprotall.iloc[:,0].values.flatten()):
        u2r_protein_finder = uniprot2reactomedf.iloc[:,0].isin([uniprotID])
        if np.any(u2r_protein_finder):
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
    if not subjectQuant.empty:
        a = int(nRuns/nGroups)
        FCcolumns = np.array(subjectQuant).astype(float)
        SEcolumns = np.tile(np.array(models.loc[:,['{SE}' in i for i in models.columns]]),(1,a)).astype(float)
        doWeights = False
    else:
        a = 1
        SEcolumns = np.array(models.loc[:,['{SE}' in i for i in models.columns]]).astype(float)
        doWeights = True

    t1 = list(np.unique(model_table['Treatment']))
    column_names = quantTableNameConstructor(t1,nRuns,isSubjectLevelQuant = False)
    PathwayQuant = pd.DataFrame(columns = ['Pathway ID','Pathway description', '# proteins','degrees of freedom','MSE']+column_names)
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
        
        abundances = FCcolumns[(ProteinsInPathway2[:,i] > 0).flatten(),:].flatten()
        treatments = np.tile(t1,(a*nprot,1)).flatten()
        SEs = np.minimum(1,abs(abundances/np.array(SEcolumns[(ProteinsInPathway2[:,i] > 0).flatten(),:]).flatten()))[:,np.newaxis]
        pathway_table = pd.DataFrame({'Protein':proteins,
                                      'Treatment':treatments})
        
        X = pd.get_dummies(pathway_table)
        parameterIDs = X.columns
        X = sp.sparse.csc_matrix(np.array(X,dtype=np.float32))
        Y = np.array(abundances)[:,np.newaxis].astype('float32')

        # Fit model
        pathwaymdl = weighted_bayeslm(X,Y,parameterIDs,doWeights,SEs,np.array([]),0)
        results = pathwaymdl['beta_estimate']
        SEMs = pathwaymdl['SEMs']
        dof = pathwaymdl['dof']
        
        Treatment_i = effectFinder(parameterIDs,'Treatment')
        Treatment_betas = list(results[Treatment_i])
        Treatment_SEMs = list(SEMs[Treatment_i])
        PathwayQuant = PathwayQuant.append(dict(zip(['Pathway ID','Pathway description', '# proteins','degrees of freedom','MSE']+column_names,[uniprot2reactomedf2.iloc[i,1],uniprot2reactomedf2.iloc[i,3],nprot2,dof,pathwaymdl['residVar']]+Treatment_betas+Treatment_SEMs+[1]*nGroups*2)),ignore_index=True,sort=False) #We'll calculate p-values (Bayes Factors?) and FDR-adjusted p-values later on.
        
        pathway_models[uniprot2reactomedf2.iloc[i,3]] = pathwaymdl
        
        timetaken = time.time()-start
        print('#',i,'/',x1,uniprot2reactomedf2.iloc[i,1],uniprot2reactomedf2.iloc[i,3],nprot2,dof,Treatment_betas, 'Took {:.2f} minutes.'.format(timetaken/60))            

    return PathwayQuant, pathway_models, uniprot2reactomedf

# Modify SE estimates and degrees of freedom as per Empirical Bayes (Smyth 2004) 
def EBvar(models):

    SEcolumns = np.array(models.iloc[:,['{SE}' in i for i in models.columns]])
    nProteins, nGroups = SEcolumns.shape
    #d0 = [0]*nGroups
    #s0 = [0]*nGroups
    DoFs = np.array(models['degrees of freedom'])
    EBDoFs = dc(DoFs)
    
    d0, null, s0 = sp.stats.chi2.fit(SEcolumns.flatten())
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
def formatData(normpeplist, exppeplist, organism, othermains_bysample = '',othermains_bypeptide = '', normmethod='median', ProteinGrouping=False, scorethreshold=0.2, nDB=1, regression_method = 'protein', ContGroup=[]):

    #get uniprot info
    print('Getting Uniprot data for',organism)
    uniprotall, upcol = getUniprotdata(organism)
    uniprotall_reviewed = uniprotall.loc[uniprotall.iloc[:,-1] == 'reviewed',:]
    print('Done!')

    #import peptide lists
    e_peplist,GroupNames,runIDs,RA,nEntries,nRuns = Progenesis2BENP(exppeplist)
    if normpeplist != exppeplist:
        #print('Importing peptide lists:', normpeplist, exppeplist)
        n_peplist = Progenesis2BENP(normpeplist)[0]
    else:
        n_peplist = dc(e_peplist)

    # Handle specification for continuous treatment variable
    if ContGroup:
        GroupIDs = set(['Continuous variable'])
        nGroups = 2
    else:
        GroupIDs = set(GroupNames)
        nGroups = len(GroupIDs)

    if othermains_bysample != '':
        e_mainsampleeffects = pd.read_csv(othermains_bysample) #.csv assigning additional column varibles to each run
        othermains_bysample_names = e_mainsampleeffects.columns
        nSmains = e_mainsampleeffects.shape[1]
    if othermains_bypeptide != '':
        e_mainpeptideeffects = pd.read_csv(othermains_bypeptide) #.csv assigning additional row varibles to each peptide
        othermains_bypeptide_names = e_mainpeptideeffects.columns
        nPmains = e_mainpeptideeffects.shape[1]

    #header = e_peplist.iloc[0:1,:];
    e_length = e_peplist.shape[0]
    scores = e_peplist.iloc[2:,7]
    scores[scores == '---'] = np.nan #Progenesis-specific
    #print(np.array(scores, dtype=float))
    scorebf = (np.log10(1/(20*(e_length - 2))) * -10) - 13 #Specific to Mascot scores
    #print(scorebf)
    pepID_p = 10**(np.array(scores, dtype=float)/-10) #Transform Mascot scores back to p-values
    pepID_fdr = bhfdr(pepID_p)
    #e_peplist = pd.concat([header,e_peplist.iloc[3:,:].loc[pepID_fdr < scorethreshold,:]])
    if othermains_bypeptide != '':
        e_mainpeptideeffects = pd.concat([e_mainpeptideeffects,e_peplist['Unnamed: 10']],axis=1)
        e_mainpeptideeffects = e_mainpeptideeffects.loc[pepID_fdr < scorethreshold,:]
        e_mainpeptideeffects = e_mainpeptideeffects.sort_values(by=['Unnamed: 10'])
        print(e_mainpeptideeffects.shape)
    
    # Initial check of reviewed status of each protein in dataset
    # Create unique (sequence and mods) id for each peptide
    e_peplist = e_peplist.iloc[2:,:].loc[pepID_fdr < scorethreshold,:]
    e_peplist = e_peplist.sort_values(by=['Unnamed: 10'])
    e_length = e_peplist.shape[0]
    gene_col = 6
    ii = 0

    while ii != e_length:
        protein_assc = e_peplist.iloc[ii,10]
        protein_find = e_peplist.iloc[:,10].isin([protein_assc])
        #protein_find.mask(protein_find == 0)
        if nDB > 1:
            protein_name = protein_assc[3:]
        else:
            protein_name = protein_assc

        uniprot_find = uniprotall.iloc[:,1].isin([protein_name])
        #uniprot_find.mask(unprot_find == 0)
        review_status = uniprotall.loc[uniprot_find,uniprotall.columns[-1]]
        protein_id = uniprotall.loc[uniprot_find,uniprotall.columns[[upcol,1]]]
        protein_sequence = uniprotall.loc[uniprot_find,uniprotall.columns[10]]
        e_peplist.loc[protein_find,e_peplist.columns[1]] = e_peplist.loc[protein_find,e_peplist.columns[[8,9]]].astype(str).sum(axis=1)
        e_peplist.loc[protein_find,e_peplist.columns[2]] = e_peplist.loc[protein_find,e_peplist.columns[gene_col+int(not ProteinGrouping)*4]]
        if protein_sequence.shape[0] > 0:
            e_peplist.loc[protein_find,e_peplist.columns[3]] = np.tile(protein_sequence,(int(np.sum(protein_find)),1)).flatten()
        else:
            e_peplist.loc[protein_find,e_peplist.columns[3]] = np.tile([''],(int(np.sum(protein_find)),1)).flatten()
        
        if len(protein_id.index):
            e_peplist.loc[protein_find,e_peplist.columns[[gene_col,10]]] = np.tile(protein_id.iloc[0,:],(int(np.sum(protein_find)),1))
            review_status = 'reviewed'
        else:
            e_peplist.loc[protein_find,e_peplist.columns[[gene_col,10]]] = protein_name
            review_status = 'unreviewed'

        e_peplist.iloc[ii:,0] = review_status
        print('#',ii,'-',ii+int(np.sum(protein_find)),'/',e_length, protein_name, review_status)
        ii = ii + int(np.sum(protein_find))

    #e_peplist = pd.concat([header,e_peplist.iloc[2:,:].sort_values(by=['Unnamed: 7'])])
    e_peplist = e_peplist.sort_values(by=['Unnamed: 7'])

    # Do normalisation to median intensities of specified peptides
    e_peplist.iloc[:,RA:] = np.log2(e_peplist.iloc[:,RA:].astype(float))
    norm_intensities = np.log2(n_peplist.iloc[2:,RA:].astype(float))
    if normmethod == 'median':
        normed_peps = normalise(e_peplist.iloc[:,RA:].astype(float),norm_intensities)
        normed_peps[np.isinf(normed_peps)] = np.nan
    else:
        print('No normalisation used!')
        normed_peps[np.isinf(normed_peps)] = np.nan
    e_peplist.iloc[:,RA:] = np.array(normed_peps)

    unique_pep_ids = np.unique(e_peplist.iloc[:,8])
    nUniquePeps = len(unique_pep_ids)
    q = 0
    final_peplist = dc(e_peplist)
    if othermains_bypeptide != '':
        final_pepmaineffectlist = dc(e_mainpeptideeffects)
    print(nUniquePeps, 'unique (by sequence) peptides.') 
    
    # Setup protein matrix if whole dataset regression is to be performed
    if regression_method == 'dataset':
        protein_columns = np.concat((pd.DataFrame(np.tile('Protein_',(uniprotall.shape[0],1))),uniprotall.iloc[:,0]),axis=1).astype(str).sum(axis=1)
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
        if np.any(pep_info.iloc[:,0].isin(['unreviewed'])):
            reviewed_prot_find = uniprotall_reviewed['Sequence'].str.contains(pep_info.iloc[0,8]) #Find all reviewed proteins that peptide could be part of
            reviewed_prot_findidx = np.where(reviewed_prot_find)[0]
            if reviewed_prot_findidx.any():
                new_protein_ids = uniprotall_reviewed.loc[reviewed_prot_find,uniprotall_reviewed.columns[[1+int(ProteinGrouping)*10,upcol]]]
                new_protein_seq = uniprotall_reviewed.loc[reviewed_prot_find,uniprotall_reviewed.columns[10]]
                is_present = np.zeros([e_length,new_protein_ids.shape[0]])
                for ii in range(new_protein_ids.shape[0]):
                    #Reassign to proteins already in dataset if possible
                    is_present[:,ii] = e_peplist.iloc[:,gene_col+int(not ProteinGrouping)*4].isin([new_protein_ids.iloc[ii,int(ProteinGrouping)]])
                is_present = np.sum(is_present,axis=0)
                ia = np.argmax(is_present)
                ic = np.amax(is_present)
                #If peptide could be shared between 2 or more equally likely reviewed proteins, skip reassignment.
                if ic != np.amin(is_present) or is_present.size == 1:
                    pep_info.loc[:,pep_info.columns[[10,6]]] = np.array(np.tile(new_protein_ids.iloc[ia,:],(nP,1)))
                    pep_info.iloc[:,0] = ['reviewed']*nP
                    pep_info.iloc[:,3] = np.array(np.tile(new_protein_seq,(nP,1)))
                    final_peplist.loc[pep_find,final_peplist.columns[[10,gene_col]]] = np.array(np.tile(new_protein_ids.iloc[ia,:],(nP,1)))
        
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
        npeps2prots = protein_matrix.sum(axis=1)    
        protein_matrix.drop(protein_matrix.columns[npeps2prots == 0],axis=1)
        q = final_peplist.shape[0]+2
        
    final_peplist = final_peplist.iloc[0:q-1,:]
    final_peplist = final_peplist.sort_values(by=['Unnamed: 2'])
    if othermains_bypeptide != '':
        final_pepmaineffectlist = final_pepmaineffectlist.iloc[0:q-1,:]
        final_pepmaineffectlist = final_pepmaineffectlist.sort_values(by=['Unnamed: 10'])
    missing_idx = np.isnan(final_peplist.iloc[:,RA:].astype(float))

    # Setup long-form table for fitting regression model
    ## Create vector of group names or continuous variable if ContGroup
    if ContGroup:
        Groups = np.tile(np.array(ContGroup),(1,(q-1))).flatten(order='F')
    else:
        Groups = np.tile(np.array(GroupNames),(1,(q-1))).flatten(order='F')

    ## Sort out new main effects, create long vectors of other variables
    if othermains_bypeptide != '':
        #e_maineffects_peptide = np.zeros(((q-1)*len(runIDs),nPmains))
        e_maineffects_peptide = np.tile(np.array(['a']),((q-1)*len(runIDs),nPmains))
        for i in range(nPmains):
            e_maineffects_peptide[:,i] = np.tile(np.array(final_pepmaineffectlist.iloc[:,i]),(len(runIDs),1)).flatten(order='F')
        e_maineffects_peptide = pd.DataFrame(e_maineffects_peptide,columns = othermains_bypeptide_names)
    if othermains_bysample != '':
        #e_maineffects_sample = np.zeros(((q-1)*len(runIDs),nSmains))
        e_maineffects_sample = np.tile(np.array(['a']),((q-1)*len(runIDs),nSmains))
        for i in range(nSmains):
            e_maineffects_sample[:,i] = np.tile(np.array(e_mainsampleeffects.iloc[:,i]),(1,(q-1))).flatten(order='F')
        e_maineffects_sample = pd.DataFrame(e_maineffects_sample,columns = othermains_bysample_names)
    if othermains_bypeptide != '' and othermains_bysample != '':
        othermains = pd.concat([e_maineffects_sample,e_maineffects_peptide],axis=1,sort=False)
    elif othermains_bypeptide != '':
        othermains = e_maineffects_peptide
    elif othermains_bysample != '':
        othermains = e_maineffects_sample

    # Subject (or run) effects
    subject = np.tile(np.array(runIDs),(1,(q-1))).flatten(order='F')

    # Peptide effects
    Peptides = np.tile(np.array(final_peplist.loc[:,final_peplist.columns[1]]),(len(runIDs),1)).flatten(order='F')

    # Get peptide and protein sequences for counting unique (by sequence) peptides for each protein and locating PTMs
    PeptideSequence = np.tile(np.array(final_peplist.loc[:,final_peplist.columns[8]]),(len(runIDs),1)).flatten(order='F')
    ProteinSequence = np.tile(np.array(final_peplist.loc[:,final_peplist.columns[3]]),(len(runIDs),1)).flatten(order='F')

    # FDR-adjusted Mascot (or other metric of peptide identity confidence) scores
    Scores = np.tile(np.array(final_peplist.loc[:,final_peplist.columns[7]].astype(float)/scorebf),(len(runIDs),1)).flatten(order='F')
    Scores[Scores > 1] = 1

    Proteins = np.tile(np.array(final_peplist.iloc[:,10+int(ProteinGrouping)*4]),(len(runIDs),1)).flatten(order='F')

    Intensities = np.array(final_peplist.iloc[:,RA:].astype(float)).flatten(order='C')

    model_table = pd.DataFrame({'Protein':Proteins,
                                'Peptide':Peptides,
                                'PeptideSequence':PeptideSequence,
                                'ProteinSequence':ProteinSequence,
                                'Score':Scores,
                                'Treatment':Groups,
                                'Subject':subject,
                                'Intensities':Intensities},
                                index = list(range(q*len(runIDs)-len(runIDs))))
        
    if othermains_bysample != '' or othermains_bypeptide != '':
        model_table = pd.concat([othermains,model_table],axis=1,sort = False)      
    
    if regression_method == 'dataset':
        protein_matrix = np.tile(protein_matrix,(nRuns,1))
        model_table = pd.concat([protein_matrix,model_table],axis=1,sort = False)  
        
    return final_peplist, model_table, missing_idx, uniprotall, nGroups, nRuns

# Model fitting function; fits individual models for each protein
def fitProteinModels(model_table,otherinteractors,incSubject,subQuantadd,nGroups,nRuns,pepmin):

    unique_proteins = np.unique(model_table.loc[:,'Protein'])
    t1 = list(np.unique(model_table.loc[:,'Treatment']))
    t2 = ['Treatment_']*len(t1)
    t = [m+str(n) for m,n in zip(t2,t1)]
    nProteins = len(unique_proteins)
    nInteractors = len(otherinteractors)
    
    #Preallocate dataframes to be filled by model fitting
    column_names = quantTableNameConstructor(t1,nRuns,isSubjectLevelQuant = False)
    ProteinQuant = pd.DataFrame(columns = ['Protein','# peptides','degrees of freedom','MSE']+column_names)
    PTMQuant = pd.DataFrame(columns=['Peptide #','Parent protein','Peptide','Scaled peptide score','PTMed residue','PTM type','PTM position in peptide','PTM position in protein','degrees of freedom']+column_names)
    models = {} #Dictionary with all fitted protein models
    #PTMQuant = pd.concat([PTMQuant,pd.DataFrame(np.zeros((ntotalPTMs,nGroups*4)),columns = column_names)],axis=1,ignore_index=True,sort=False)
    
    # Summing user-specified effects for subject-level protein quantification
    SubjectLevelColumnNames = quantTableNameConstructor(t1,nRuns,isSubjectLevelQuant = True)
    SubjectLevelProteinQuant = pd.DataFrame(columns = SubjectLevelColumnNames)
    SubjectLevelPTMQuant = pd.DataFrame(columns = SubjectLevelColumnNames)
    
    #Fit protein models
    print('Fitting protein models...')
    q = 0
    #v = 0
    for protein in unique_proteins:
        start = time.time()
        protein_table = model_table.loc[model_table.loc[:,'Protein'] == protein,:].reset_index(drop=True)
        nPeptides = len(np.unique(protein_table['PeptideSequence']))#1+np.sum(Peptide_i.astype(int))
       
        #if protein != 'XPP1_HUMAN' and v == 0:
        #    continue
        if nPeptides < pepmin:
            continue #Skip proteins with fewer than pepmin peptides in dataset
        #v = 1
        #Create design matrix (only add Treatment:Peptides interactions if more than 2 peptides)
        X = designMatrix(protein_table,otherinteractors,incSubject,len(np.unique(protein_table['Peptide'])))
        Y = np.array(protein_table.loc[:,'Intensities'])[:,np.newaxis].astype('float32')
        parameterIDs = X.columns
        p = ['Peptide_'+str(n) for n in list(np.unique(protein_table.loc[:,'Peptide']))]
        X_missing = X[parameterIDs.intersection(t+p)]
        missing = np.isnan(Y)
        Y_missing = (np.sign(missing.astype(float)-0.5)*10).astype('float32')
        parameterIDs_missing = X_missing.columns
        n = X_missing.shape[0]
        X_missing = sp.sparse.csc_matrix(np.array(X_missing,dtype=np.float32))
        X = sp.sparse.csc_matrix(np.array(X,dtype=np.float32))
        
        # Fit missing model to choose which missing values are MNR and which are MAR
        missingmdl = weighted_bayeslm(X_missing,Y_missing,parameterIDs_missing,False,np.ones([n,1]),np.array([]),nInteractors)
        MNR = np.ravel(np.sum(np.concatenate((missingmdl['beta_estimate'] > 0, missingmdl['beta_estimate'] > missingmdl['b0']),axis=0),axis=0)) == 2
        Y_MNR = (X_missing[np.where(missing)[0],:][:,MNR].sum(axis=1) > 0)

        # Fit protein model to estimate fold changes while imputing missing values
        proteinmdl = weighted_bayeslm(X,Y,parameterIDs,True,np.array(protein_table['Score'])[:,np.newaxis],Y_MNR,nInteractors)
        #b0SEM = list(proteinmdl['b0SEM'])
        results = proteinmdl['beta_estimate']
        SEMs = proteinmdl['SEMs']
        dof = proteinmdl['dof']

        # Sort out protein-level effects
        Treatment_i = effectFinder(parameterIDs,'Treatment')
        Treatment_betas = list(results[Treatment_i])
        Treatment_SEMs = list(SEMs[Treatment_i])
        ProteinQuant = ProteinQuant.append(dict(zip(['Protein','# peptides','degrees of freedom','MSE']+column_names,[protein,nPeptides,dof,proteinmdl['residVar']]+Treatment_betas+Treatment_SEMs+[1]*nGroups*2)),ignore_index=True,sort=False) #We'll calculate p-values (Bayes Factors?) and FDR-adjusted p-values later on.

        # Sort out treatment:peptide interaction effects
        TreatmentPeptide_i = effectFinder(parameterIDs,'Treatment',True,'Peptide')
        TreatmentPeptide_names = parameterIDs[np.newaxis][TreatmentPeptide_i]
        TreatmentPeptide_betas = results[TreatmentPeptide_i]
        TreatmentPeptide_SEMs = SEMs[TreatmentPeptide_i]

        ntotalPTMs = 0
        for peptide in np.unique(protein_table.loc[:,'Peptide']):
            PTMpositions_in_peptide = np.array(re.findall('\[([0-9]+)\]',peptide)).astype(int)-1 #PTM'd residues denoted by [#]
            if len(PTMpositions_in_peptide) > 0:
                #Get rows with mod, modded residue, modded peptideID for each peptide
                PTMdResidues = [peptide[i] for i in PTMpositions_in_peptide]
                PTMs = re.findall('(?<=] )\S+',peptide)
                peptide_finder = protein_table.loc[:,'Peptide'].isin([peptide])
                parent_protein_sequnce = protein_table.loc[peptide_finder,'ProteinSequence'].iloc[0]
                peptide_sequence = protein_table.loc[peptide_finder,'PeptideSequence'].iloc[0]
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
                peptide_score = np.mean(np.array(protein_table.loc[peptide_finder,'Score'])) #Uses mean Mascot score of all instances of that peptide, for MaxQuant use localisation score (FDR filter peptides beforehand and give unmodified peptides an arbitrarily high value?)?
                
                #Get effect value and SEM for modded peptide:treatment interaction
                beta_finder = list(filter(lambda x: x.endswith(peptide), list(TreatmentPeptide_names))) #[beta.start() for beta in re.finditer('Peptide_'+peptide+'$',list(TreatmentPeptide_names))
                PTMbetas = TreatmentPeptide_betas[[list(TreatmentPeptide_names).index(beta) for beta in beta_finder]] #[TreatmentPeptide_betas[i] for i in beta_finder]

                if PTMbetas.size == 0:
                    continue
                PTMSEMs = TreatmentPeptide_SEMs[[list(TreatmentPeptide_names).index(beta) for beta in beta_finder]] #[TreatmentPeptide_SEMs[i] for i in beta_finder]
                PTMvalues = list(PTMbetas)+list(PTMSEMs)+[1]*nGroups*2

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
                    PTMrow.update(dict(zip(column_names,PTMvalues)))
                        
                    PTMQuant = PTMQuant.append(PTMrow,ignore_index=True,sort=False)
                    
                    if subQuantadd != ['']:
                        # Subject-level PTM quantification by summing Treatment:Peptide interactions with user-specified:Peptide interaction terms
                        #subjectPTMQuant = np.concatenate((np.zeros((1,1)),results[TreatmentPeptide_i][np.newaxis]),axis=1).T
                        subjectPTMQuant = results[TreatmentPeptide_i][np.newaxis].T
                        for parameter in subQuantadd:
                            subjectPTMQuant_i = effectFinder(parameterIDs,re.escape('Peptide_'+peptide),True,re.escape(parameter))+effectFinder(parameterIDs,re.escape(parameter),True,re.escape('Peptide_'+peptide))
                            subjectPTMQuant_betas = np.tile(results[subjectPTMQuant_i][np.newaxis],(subjectPTMQuant.size,1))
                            subjectPTMQuant = subjectPTMQuant + subjectPTMQuant_betas
                            subjectPTMQuant = np.reshape(subjectPTMQuant,(-1,1),order='F')
                            SubjectLevelPTMQuant = SubjectLevelPTMQuant.append(dict(zip(SubjectLevelColumnNames,subjectPTMQuant)),ignore_index=True,sort=False)

            ntotalPTMs = ntotalPTMs + len(PTMpositions_in_peptide)
        
        if subQuantadd != ['']:
            #Sort out Subject-level protein quantification
            subjectLevelQuant = results[Treatment_i][np.newaxis].T
            for parameter in subQuantadd:
                subjectQuant_i = effectFinder(parameterIDs,parameter)
                subjectQuant_betas = np.tile(results[subjectQuant_i][np.newaxis],(subjectLevelQuant.size,1))
                subjectLevelQuant = subjectLevelQuant + subjectQuant_betas
                subjectLevelQuant = np.reshape(subjectLevelQuant,(-1,1),order='F')
                SubjectLevelProteinQuant = SubjectLevelProteinQuant.append(dict(zip(SubjectLevelColumnNames,subjectLevelQuant)),ignore_index=True,sort=False)

        #Store model with all parameters
        models[protein] = proteinmdl
        timetaken = time.time()-start
        if ntotalPTMs > 0:
            print('#',q,'/',nProteins,protein,nPeptides,dof,Treatment_betas,'Found', ntotalPTMs,'PTM(s).', 'Took {:.2f} minutes.'.format(timetaken/60))# at ', [m+str(n) for m,n in zip(PTMdResidues,PTMpositions_in_protein)])
        else:
            print('#',q,'/',nProteins,protein,nPeptides,dof,Treatment_betas,'Found 0 PTM(s).', 'Took {:.2f} minutes.'.format(timetaken/60))            
        q += 1
    
    #Clean up PTM quantification to account for different peptides (missed cleavages) that possess the same PTM at same site
    PTMQuant_cleaned,SubjectLevelPTMQuant_cleaned = cleanPTMquants(PTMQuant,nGroups,SubjectLevelPTMQuant)
    
    return ProteinQuant, PTMQuant_cleaned, SubjectLevelProteinQuant, SubjectLevelPTMQuant_cleaned, models

# Use PyMC3 to fit a single model for the entire dataset - very computationally intensive
def fitDatasetModel(model_table,otherinteractors,incSubject,subQuantadd,nGroups,nRuns,pepmin):
    
    t1 = list(np.unique(model_table.loc[:,'Treatment']))
    t2 = ['Treatment_']*len(t1)
    t = [m+str(n) for m,n in zip(t2,t1)]
    nInteractors = len(otherinteractors)
    
    protein_list = np.unique(model_table.loc[:,'Protein'])
    nProteins = len(protein_list)
    
    X = designMatrix(model_table,otherinteractors,incSubject,len(np.unique(model_table['Peptide'])),regmethod='dataset')
    Y = np.array(model_table.loc[:,'Intensities'])[:,np.newaxis].astype('float32')
    parameterIDs = X.columns
    p = ['Peptide_'+str(n) for n in list(np.unique(model_table.loc[:,'Peptide']))]
    X_missing = X[parameterIDs.intersection(t+p)]
    missing = np.isnan(Y)
    Y_missing = (np.sign(missing.astype(float)-0.5)*10).astype('float32')
    parameterIDs_missing = X_missing.columns
    n = X_missing.shape[0]
    X_missing = sp.sparse.csc_matrix(np.array(X_missing,dtype=np.float32))
    X = sp.sparse.csc_matrix(np.array(X,dtype=np.float32))
    
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
    for peptide in np.unique(model_table.loc[:,'Peptide']):
        PTMpositions_in_peptide = np.array(re.findall('\[([0-9]+)\]',peptide)).astype(int)-1 #PTM'd residues denoted by [#]
        if len(PTMpositions_in_peptide) > 0:
            #Get rows with mod, modded residue, modded peptideID for each peptide
            PTMdResidues = [peptide[i] for i in PTMpositions_in_peptide]
            PTMs = re.findall('(?<=] )\S+',peptide)
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
                          'Peptide':peptide,
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

    return ProteinQuant,PTMQuant_cleaned,SubjectLevelProteinQuant,SubjectLevelPTMQuant_cleaned,models

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
            PTMrow['Parent protein'] = ptms['Parent protein'].iloc[0]
            PTMrow['PTMed residue'] = ptms['PTMed residue'].iloc[0]
            PTMrow['PTM type'] = ptms['PTM type'].iloc[0]
            PTMrow['PTM position in protein'] = ptms['PTM position in protein'].iloc[0]
            PTMSubjectrow = ptmssubject.mean(axis=0, numeric_only = True) #collapse all subject values to averages
            PTMSEs = np.sqrt((ptms.iloc[:,9+nGroups:9+nGroups*2].values**2).sum(axis=0)) #collapse summary errors by square-root of sum of squared errors
            PTMrow[['{SE}' in i for i in PTMrow.index]] = PTMSEs
            PTMQuant_cleaned = PTMQuant_cleaned.append(PTMrow, ignore_index = True, sort = False)
            SubjectLevelPTMQuant_cleaned = SubjectLevelPTMQuant_cleaned.append(PTMSubjectrow, ignore_index = True, sort = False)
        else:
            PTMrow = SubjectLevelPTMQuant.loc[ptm_finder,:]
            PTMSubjectrow = SubjectLevelPTMQuant.loc[ptm_finder,:]
            PTMQuant_cleaned = PTMQuant_cleaned.append(PTMrow, ignore_index = True, sort = False)
            SubjectLevelPTMQuant_cleaned = SubjectLevelPTMQuant_cleaned.append(PTMSubjectrow, ignore_index = True, sort = False)  
            
    return PTMQuant_cleaned, SubjectLevelPTMQuant_cleaned
 
# Create design matrix for Treatment + Peptide + Treatment*Peptide + additional user-specified main and interaction effects.
def designMatrix(protein_table,interactors,incSubject,nPeptides,regmethod = 'protein'):
    
    # Create design matrix - all fixed effects... for now.
    if incSubject:
        X_table = protein_table.loc[:,protein_table.columns.isin(['Protein','ProteinSequence','PeptideSequence','Score','Intensities'])!=True]
    else:
        X_table = protein_table.loc[:,protein_table.columns.isin(['Protein','ProteinSequence','PeptideSequence','Score','Intensities','Subject'])!=True]
    
    if regmethod == 'dataset':
        protein_effects = effectFinder(X_table, 'Protein_')
        X_main = pd.get_dummies(X_table.loc[:,protein_effects != True])
        X_main = pd.concat((X_table.loc[:,protein_effects],X_main),axis=1,sort = False)
    else:
        X_main = pd.get_dummies(X_table)
        
    X_main_labs = X_main.columns
    #print(X_main_labs)
    q = X_main.shape[0]

    # Process interactions (specified as a dictionary e.g. {Interactor1:Interactor2,Interactor3:Interactor1})
    ## Default is alway Peptide:Treatment, others are user-specified
    X_interactors = np.zeros([q,1])
    X_interactors = pd.DataFrame(X_interactors)
    for i in X_main_labs:
        if 'Treatment' in i and nPeptides > 2:
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
        pattern = '(?<!:)'+pattern+'(?!\w+:)'
        
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
        
# Median normalisation for peptide intensities to specified set of peptides
def normalise(X,Z=[]):
    if not Z.empty:
        median_Z = np.nanmedian(Z,axis=0)
    else:
        median_Z = np.nanmedian(X,axis=0)
    normalised_X = X - median_Z
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

# Pull 'species' data from Uniprot where 'species' is either 'human' or 'mouse'
def getUniprotdata(species):
    if species == 'human':
        url = 'http://www.uniprot.org/uniprot/?sort=&desc=&compress=no&query=proteome:UP000005640&fil=&force=no&preview=true&format=tab&columns=id,entry%20name,protein%20names,genes,go,go(biological%20process),go(molecular%20function),go(cellular%20component),go-id,interactor,sequence,genes(PREFERRED),reviewed'
        upcol = 11
    elif species == 'mouse':
        url = 'http://www.uniprot.org/uniprot/?sort=&desc=&compress=no&query=proteome:UP000000589&fil=&force=no&preview=true&format=tab&columns=id,entry%20name,protein%20names,genes,go,go(biological%20process),go(molecular%20function),go(cellular%20component),go-id,interactor,sequence,genes(PREFERRED),database(MGI),reviewed'
        upcol = 12
    else:
        url = 'http://www.uniprot.org/uniprot/?sort=&desc=&compress=no&query=proteome:'+species+'&fil=&force=no&preview=true&format=tab&columns=id,entry%20name,protein%20names,genes,go,go(biological%20process),go(molecular%20function),go(cellular%20component),go-id,interactor,sequence,genes(PREFERRED),reviewed'
        upcol = 12
        
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    uniprotall = response.read().decode('utf-8')

    with open("uniprotall.tab", "w") as tsv:
        print(uniprotall, file = tsv)

    updf = pd.read_table("uniprotall.tab")

    return updf, upcol

# Import peptide tables from peplist, a Progenesis QI formatted .csv file
def Progenesis2BENP(peplist):
    # import normalisation peptidelist
    print('Importing',peplist,'peptide list: ')
    with open(peplist, newline = '') as csvfile:
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
        pepdf = pd.read_csv(peplist)
        return pepdf, GroupNames, runs, RAind, n-2,nRuns

# Weighted Bayesian regression function not typically called by user
# MNR must by boolean numpy array!!
        
def weighted_bayeslm(X,Y,featureIDs,do_weights,Scores,MNR,nInteractors):
    wX = dc(X)
    [n,p] = wX.shape
    wY = dc(Y)
    Yimputed = dc(Y)
    meanY = np.nanmean(Y).flatten()
    np.random.seed(1345)
    iNumIter = 1000
    iBurn = 500

    tau_vector = np.random.rand(p)
    D_tau_squared = sp.sparse.csc_matrix(np.diag(tau_vector.flatten()))

    XtX = sp.sparse.csc_matrix(X.T @ X)
    w = np.ones((n,1),dtype=np.float32)
    sigma2_shape = (n-1+p)/2;
    beta_posterior = np.zeros((iNumIter-iBurn,p))
    intercept = np.zeros((iNumIter-iBurn,1))
    beta_estimate = np.random.randn(1,p)
    b0 = np.random.randn(1,1)
    sigma_squared = np.zeros((iNumIter-iBurn,1))
    sigma2 = 1/np.random.gamma(sigma2_shape, 0.01)
    lambda_lasso = np.array(np.sqrt(np.random.gamma(p,1,(p))))
    lambda_ridge = np.array(np.random.gamma(1,1/3,(p)))
    DoF = np.zeros((iNumIter-iBurn,1))

    # Imputation variables
    impmin = np.nanmin(wY)-2
    Ymissing = np.isnan(wY)
    Ymissing_i = np.where(Ymissing)[0]
    nMissing = Ymissing_i.size
    nMNR = int(np.sum(MNR,0))
    nMR = nMissing - nMNR
    prop_MNR = nMNR/n
    impY = np.full((nMissing,iNumIter-iBurn),np.nan)
    D = np.tile(meanY,nMR)
    if nMissing:
        alpha = np.percentile(wY[np.where(Ymissing == False)[0]],prop_MNR*100)
        if nMR:
            Xmiss = X[Ymissing_i[np.where(MNR == False)[0]],:].toarray()    # MNR must be np.array
            try:
                XXTYMR = np.linalg.inv((Xmiss @ Xmiss.T)*np.eye(nMR)) 
                np.linalg.cholesky(XXTYMR)
            except:
                nMR = False # If inv fails probably means there are too many missing values -> protein is low abundance -> use MNR imputation
                MNR = np.array([True for i in MNR])
                nMNR = nMissing
       
    ii = 0
    for i in range(iNumIter+1):
        if nMissing:
            if nMNR:
                # Impute MNR missing values from truncated gaussian
                z = sp.stats.truncnorm.rvs(impmin,alpha,loc=meanY,scale=sigma2,size=(nMNR,1))
                wY[Ymissing_i[np.where(MNR)[0]]] = z*w[Ymissing_i[np.where(MNR)[0]]]
            if nMR:
                # Impute MR missing values from multivariate_normal
                B = sigma2*XXTYMR
                #D = b0 + Xmiss @ beta_estimate.T #np.ndarray.flatten(np.concatenate((X0[Ymissing_i[np.where(MNR == False)[0]],:],Xmiss),axis=1) @ np.concatenate((b0,beta_estimate),axis=1).T.flatten())
                #D = np.array([x for y in D for x in y]).ravel() #FUCK ME, WHY IS IT IMPOSSIBLE TO FLATTEN D??????!!!!!!!!
                wY[Ymissing_i[np.where(MNR == False)[0]]] = (w[Ymissing_i[np.where(MNR == False)[0]]].T*np.random.multivariate_normal(D,B)).T
            
            Yimputed[Ymissing] = wY[Ymissing]/w[Ymissing]

        # beta_estimate from conditional multivariate_normal
        L = sp.sparse.csc_matrix(np.diag((lambda_ridge+tau_vector).ravel()))
        L = sp.sparse.linalg.inv(XtX+L)
        C = L.multiply(sigma2)
        A = np.ndarray.flatten(L @ (wX.T @ wY))
        beta_estimate = np.random.multivariate_normal(A,C.toarray())[np.newaxis]
        b0 = sigma2*np.random.randn(1,1)+np.nanmean(Yimputed)

        # sigma**2 from inverse gamma
        residuals = Yimputed - (b0 + X @ beta_estimate.T) #np.concatenate((X0,X),axis=1) @ np.concatenate((b0,beta_estimate),axis=1).T
        sigma2_scale = (residuals.T @ residuals)/2 + ((sp.sparse.linalg.inv(D_tau_squared) @ (beta_estimate*lambda_lasso).T).T @ beta_estimate.T)/2 + ((beta_estimate*lambda_ridge) @ beta_estimate.T)/2
        sigma2 = 1/np.random.gamma(sigma2_shape,1/(sigma2_scale+0.01) + 0.01) #Change to .ravel() ??
 
        # 1/tau**2 from IG
        tau2_shape = np.sqrt(lambda_lasso**2/beta_estimate**2*sigma2)
        tau2_scale = lambda_lasso**2
        tau_vector = np.random.wald(tau2_shape,tau2_scale)
        D_tau_squared = sp.sparse.csc_matrix(np.diag(tau_vector.ravel()))

        # lambda_lasso and lambda_ridge from gamma
        lambda_lasso[:,] = np.sqrt(np.random.gamma(p+nInteractors, 1+(1/tau_vector).sum()/2))
        lambda_ridge[:,] = np.random.gamma(1+nInteractors, 1/(beta_estimate**2/2/sigma2+3)+0.01)

        if i > iBurn:
            beta_posterior[ii,:] = beta_estimate
            DoF[ii] = np.sum((X @ L @ X.T).diagonal())
            intercept[ii] = b0
            sigma_squared[ii] = sigma2
            impY[:,ii] = Yimputed[Ymissing]
            ii = ii + 1

        if do_weights:
            r = 1/(0.5+residuals**2/2/sigma2)
            s = np.random.binomial(1,Scores)
            w = 1+s+np.random.gamma(s+0.5, r+0.01)
            wY = Yimputed*w
            wX = X.multiply(w)
            XtX = sp.sparse.csc_matrix(wX.T @ wX)

    impY = np.nanmean(impY,1)
    Yimputed[Ymissing] = impY

    beta_SEMs = np.std(beta_posterior,axis=0)
    beta_estimate = np.mean(beta_posterior,axis=0)
    b0 = np.mean(intercept,axis=0)
    b0SEM = np.std(intercept,axis=0)
    sigma2 = np.mean(sigma_squared)
    yfit = X @ beta_estimate.T
    #tscores = np.abs(beta_estimate)/SEMs
    #dof = beta_estimate[beta_estimate!=0].size
    dof = np.mean(DoF)
    #pvalues = (1 - sp.stats.t.cdf(tscores,dof)) * 2
    
    return {'beta_posterior':beta_posterior,
            'b0_posterior':intercept,
            'beta_estimate':beta_estimate[np.newaxis],
            'b0':b0,
            'SEMs':beta_SEMs[np.newaxis],
            'b0SEM':b0SEM,
            'Yimputed':Yimputed,
            'residVar':sigma2,
            'yfit':yfit,
            'dof':dof,
            'parameters':featureIDs}#,

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
    w = np.ones((n,1),dtype=np.float32)
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
            s = np.random.binomial(1,Scores)#s = pm.Binomial('s',n=1,p=scores,shape=(n,1))
            w = 1+s+pm.Gamma('w',alpha=0.5+s, beta=r,shape=(n,1))
            wY = Yimputed*w
            wX = X*w
            XtX = T.dot(wX.T,wX)
            
        Y_obs = pm.MvNormal('Y_obs', mu=b0+T.dot(X,beta_estimate.T), cov=sigma2, observed=Yimputed,shape=(n,1))
        trace = pm.sample(500,njobs=1,nchains=2,nuts_kwargs={'max_treedepth':20,'target_accept':0.6,'integrator':'three-stage'},n_init=200,init='advi+adapt_diag')
        
    return bayes_lm,trace

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