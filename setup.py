from setuptools import setup, find_packages

setup(
    name="sygnals-nn",
    version="1.6.0", # Bump version for new features
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "pandas",
        "numpy",
        "click",
        "tf2onnx",
        "scikit-learn",
        "joblib",
        "nltk",
        "sentence-transformers",
        "onnxruntime",
        "tensorflow-probability" # Added TFP
    ],
    entry_points={
        "console_scripts": [
            "sygnals-nn=sygnals_nn.cli:cli"
        ]
    },
    python_requires='>=3.11', # Specify Python version compatibility
    author="Araray Velho", # Example author
    author_email="araray@example.com", # Example email
    description="A CLI tool for creating, training, and managing neural networks, including probabilistic models.", # Updated description
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/araray/sygnals-nn", # Example URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Example license
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta", # Example status
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
