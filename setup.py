from setuptools import setup, find_packages
from pathlib import Path

version = Path("VERSION").read_text().strip()

setup(
    name="nicegold_enterprise",
    version=version,
    description="NICEGOLD Enterprise: XAUUSD M1 trading pipeline",
    packages=find_packages(where="src", exclude=["tests", "docs"]),
    package_dir={"": "src"},
    install_requires=[
        "pandas==2.2.2",
        "numpy==1.26.4",
        "scikit-learn==1.6.1",
        "catboost==1.2.8",
        "ta==0.11.0",
        "optuna>=3.0.0",
        "shap>=0.41.0",
        "psutil>=5.9.0",
    ],
    python_requires=">=3.8,<3.11",
    entry_points={
        "console_scripts": [
            "nicegold=main:main",
            "nicegold-sweep=tuning.hyperparameter_sweep:main",
        ]
    },
    include_package_data=True,
    license="MIT",
    author="Phiradon168",
    url="https://github.com/Phiradon168",
)
