"""
Setup and packaging configuration for kut_geometry_engine.
Axiom: E=C (Equivalence of Computation and Topological Information Energy)
"""

from setuptools import setup, find_packages

setup(
    name="kut-geometry-engine",
    version="1.1.0",
    description="Test-Time Search Optimization via Geometric Ricci Annealing and Multi-Attractor KUT",
    author="Junki Kanamori / KUT Research Group",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Mathematics",
    ],
)
