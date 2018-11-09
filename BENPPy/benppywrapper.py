# -*- coding: utf-8 -*-
"""
BENPPy wrapper for testing

Created on Thu Nov  8 14:33:13 2018

@author: Venk
"""
import BayesENproteomics as bp

a = bp.BayesENproteomics(output_name='test',form='progenesis')

a.doAnalysis('testpeplist2.csv', 
              'testpeplist.csv',  
              'human',               
              othermains_bysample = 'testsamplemains.csv',    
              otherinteractors = {'Peptide':'Donor'},       
              regression_method = 'dataset',
              subQuantadd = ['Donor'],
              form = 'progenesis')

a.doContrasts(Contrasted = 'protein',
              ctrl = 0, 
              propagateErrors = True,
              UseBayesFactors = False)

a.Contrasted.to_csv(a.output_name+'\\Contrasted.csv',
                    encoding = 'utf-8',
                    index = False,
                    header = a.Contrasted.columns)

