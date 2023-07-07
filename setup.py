from setuptools import setup, find_packages

setup(
    name='crossval_ensemble',
    version='0.0.1',
    description='A scikit-learn wrapper for CrossValidation Ensembles',
    long_description='CrossvalEnsemble is an ML library in Python that allows to create\
    CrossValidation Ensembles leveraging scikit-learn.',
    long_description_content_type='text/markdown',
    author='Pierre-Alexis Thoumieu',
    author_email='pierre-alexis@liberkeys.com',
    packages=find_packages(exclude=['tests']),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.25.0',
        'pandas>=2.0.3',
        'tqdm>=4.65.0',
        'python-dotenv>=1.0.0',
        'flake8>=6.0.0',
        'scikit-learn>=1.3.0',
        'ipywidgets>=8.0.6',
        'catboost>=1.2',
        'xgboost>=1.7.6',
        'lightgbm>=3.3.5',
        'sklearn-pandas>=2.2.0',
    ]
)
