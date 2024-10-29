from setuptools import find_packages, setup

setup(
    name="GBOpt",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'numba',
        'matplotlib',
        'pytest'
    ],
    author="Chaitanya Bhave and Jarin French",
    author_email="chaitanya.bhave@inl.gov and jarin.french@inl.gov",
    description="A package for grain boundary optimization",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.inl.gov/chaitanya-bhave/GBOpt",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
        "Operating System :: OS Independent",
    ],
)
