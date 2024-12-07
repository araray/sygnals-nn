from setuptools import setup, find_packages

setup(
    name="sygnals-nn",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "pandas",
        "numpy",
        "click"
    ],
    entry_points={
        "console_scripts": [
            "sygnals-nn=sygnals_nn.cli:cli"
        ]
    },
)

