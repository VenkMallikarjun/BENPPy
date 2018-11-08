import setuptools

def readme():
    with open('README.md') as readme_file:
        return readme_file.read()

configuration = {
    'name' : 'BENPPy',
    'version': '1.0.5',
    'description' : 'BayesENproteomics in Python',
    'long_description' : readme(),
    'classifiers' : [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
    ],
    'keywords' : 'proteomics PTMs MCMC',
    'url' : 'https://github.com/VenkMallikarjun/BENPPy',
    'maintainer' : 'Venkatesh Mallikarjun',
    'maintainer_email' : 'vjmallikarjun@gmail.com',
    'license' : 'MIT',
    'packages' : ['BENPPy'],
    'install_requires': ['pymc3']
    }

setuptools.setup(**configuration)
