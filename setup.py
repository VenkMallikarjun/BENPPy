import setuptools

def readme():
    with open('README.md') as readme_file:
        return readme_file.read()

configuration = {
    'name' : 'BENPPy',
    'version': '2.4.2',
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
    'author' : 'Venkatesh Mallikarjun',
    'author_email' : 'vjmallikarjun@gmail.com',
    'license' : 'MIT License',
    'packages' : ['BENPPy'],
    'install_requires': ['pymc3']
    }

setuptools.setup(**configuration)
