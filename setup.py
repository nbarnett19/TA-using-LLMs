from setuptools import setup, find_packages

setup(
    name="TA-using-LLMs",
    version="0.0.1",
    authors="Natalie Barnett",
    author_email="nataliebarnett.ch@gmail.com",
    description="An application that performs qualitative thematic analysis using LLMs",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entrypoints={"console_scripts": ["TA_using_LLMs = src.main:main"]},
    homepage="https://github.com/nbarnett19/TA-using-LLMs",
)