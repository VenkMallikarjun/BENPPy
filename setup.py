import setuptools

def readme():
    with open('README.md') as readme_file:
        return readme_file.read()

configuration = {
    'name' : 'BENPPy',
    'version': '2.6.4',
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
    'install_requires': ['numpy','scipy','sklearn','matplotlib','pandas']
    }

setuptools.setup(**configuration)
