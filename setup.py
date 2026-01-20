# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from setuptools import find_packages, setup

setup(
    name="GBOpt",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy<=2.1",
        "scipy",
        "numba",
        "pandas",
        "matplotlib",
        "pytest",
        "spglib"
    ],
    author="Chaitanya Bhave and Jarin French",
    author_email="chaitanya.bhave@inl.gov and jarin.french@inl.gov",
    description="A package for grain boundary optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/IdahoLabResearch/GBOpt",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT",
        "Operating System :: OS Independent",
    ],
)
