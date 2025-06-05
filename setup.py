from setuptools import setup, find_packages

setup(
    name='nicegold_enterprise',
    version=open('VERSION').read().strip(),
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas>=2.2.2',
        'numpy>=2.0.2',
        'scikit-learn>=1.6.1',
        'ta>=0.11.0',
        'optuna>=3.0.0',
        'catboost>=1.0.6',
        'psutil>=5.9.0',
        'shap>=0.41.0',
        'GPUtil>=1.4.0',
        'torch>=2.0.0',
    ],
    python_requires='>=3.8,<3.11',
    entry_points={
        'console_scripts': [
            'nicegold-main = src.main:main',
            'nicegold-sweep = tuning.hyperparameter_sweep:main'
        ]
    }
)
