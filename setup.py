from setuptools import setup, find_packages

setup(
    name='subspace_median',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=1.8'
    ],
    description='A package for computing subspace median using Weiszfeld\'s algorithm.',
    author='Ankit Pratap Singh',
    author_email='sankit@iastate.edu',
    url='https://github.com/singhankitpratap/subspace_median',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)