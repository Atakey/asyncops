#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
@Author: huanghailong
@contact: huanghailong@xiaoyouzi.com
@time: 2024/9/1 0:10
@desc: 
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="asyncops",
    version="0.1.0",
    author="Huanghl",
    author_email="huanghailong@xiaoyouzi.com",
    description="A package for asynchronous operations with pandas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Atakey/asyncops",
    packages=find_packages(
        exclude=['*test*'],
    ),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.0.0",
        "psutil>=5.0.0",
        "pandarallel>=1.5.0",
        "cache_to_disk>=2.0.0",
    ],
)
