#setup.py
from setuptools import setup, find_packages

setup(
    name="TA_using_LLMs",
    version="0.0.2",
    authors="Natalie Barnett",
    author_email="nataliebarnett.ch@gmail.com",
    description="An application that performs qualitative thematic analysis using LLMs",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entrypoints={"console_scripts": ["TA_using_LLMs = src.TA_using_LLMs.main:main"]},
    homepage="https://github.com/nbarnett19/Thematic_Analysis_using_LLMs",
)