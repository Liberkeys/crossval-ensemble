from setuptools import setup, find_packages

setup(
    name='crossval_ensemble',
    version='0.0.0',
    description='A scikit-learn wrapper for CrossValidation Ensembles',
    long_description='CrossvalEnsemble is an ML library in Python that allows to create\
    CrossValidation Ensembles leveraging scikit-learn.',
    long_description_content_type='text/markdown',
    author='Pierre-Alexis Thoumieu',
    author_email='pierre-alexis@liberkeys.com',
    packages=find_packages(exclude=['tests']),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'tqdm',
        'python-dotenv',
        'flake8',
        'scikit-learn',
        'ipywidget',
        'catboost',
        'xgboost',
        'lightgbm',
        'sklearn-pandas',
        'plotly'
    ],
    project_urls={
        'Documentation': '',
        'Source': '',
        'Tracker': '',
    }
)
