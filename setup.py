from setuptools import setup, find_packages

setup(
    name="P2P",
    version="0.1.0",
    description="A Predictive-to-Prescriptive Framework for Supply Chain Analytics",
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.15.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "PuLP>=2.7.0",
        "pyyaml>=6.0",
    ],

    # Everything needed for ml & testing (pip install -e ".[dev,ml]")
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.1",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
            "ipython>=8.14.0",
            "ipywidgets>=8.0.7",
        ],
        "ml": [
            "torch>=2.0.0",
            "neuralforecast>=1.6.0",
        ],
    },
)