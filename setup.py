from setuptools import setup

def readme():
    with open('README.md') as readme_file:
        return readme_file.read()

configuration = {
    'name' : 'BENPPy',
    'version': '1.0',
    'description' : 'BayesENproteomics in Python',
    'long_description' : readme(),
    'classifiers' : [
        'Development Status :: Alpha',
        'Intended Audience :: Science/Research',
        'License :: MIT',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.6',
    ],
    'keywords' : 'proteomics PTMs MCMC',
    'url' : 'https://github.com/VenkMallikarjun/BENPPy',
    'maintainer' : 'Venkatesh Mallikarjun',
    'maintainer_email' : 'vjmallikarjun@gmail.com',
    'license' : 'BSD',
    'packages' : ['BENPPy'],
    'install_requires': ['pymc3'],
    'ext_modules' : [],
    'cmdclass' : {},
    'test_suite' : 'nose.collector',
    'tests_require' : ['nose'],
    'data_files' : ()
    }

setup(**configuration)
