from setuptools import setup, find_packages

setup(
    name="pay-attention-pipeline",
    version="0.1.0",
    author="Pedro Silva",
    author_email="pedrolmssilva@gmail.com",
    description="A Python package that uses PyTorch, Transformers, and NumPy",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/netop-team/pay_attention_pipeline",  # Replace with your GitHub repo or project URL
    packages=find_packages(),
    install_requires=[
        "torch>=1.0.0",
        "transformers>=4.0.0",
        "numpy>=1.18.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)