import setuptools

def readme():
    with open('README.md') as readme_file:
        return readme_file.read()

configuration = {
    'name' : 'BENPPy',
    'version': '2.6.5',
    'description' : 'BayesENproteomics in Python',
    'long_description' : readme(),
    'classifiers' : [
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
    ],
    'keywords' : 'proteomics PTMs MCMC',
    'url' : 'https://github.com/VenkMallikarjun/BENPPy',
    'author' : 'Venkatesh Mallikarjun',
    'author_email' : 'vjmallikarjun@gmail.com',
    'license' : 'MIT License',
    'packages' : ['BENPPy'],
    'install_requires': ['numpy','scipy','scikit-learn','matplotlib','pandas']
    }

setuptools.setup(**configuration)
