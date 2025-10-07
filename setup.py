#!/usr/bin/env python3
"""
Setup script for The True Experience – AI Film Analysis System
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-film-analysis",
    version="1.0.0",
    author="AI Film Analysis Team",
    author_email="contact@ai-film-analysis.com",
    description="A comprehensive video analysis system that generates detailed, timestamped JSON summaries",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/ai-film-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "gpu": [
            "torch[cuda]>=2.0.0",
            "torchvision[cuda]>=0.15.0",
            "torchaudio[cuda]>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-film-analysis=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
    keywords=[
        "video analysis",
        "AI",
        "machine learning",
        "computer vision",
        "audio processing",
        "multimodal",
        "film analysis",
        "scene detection",
        "emotion analysis",
        "dialogue processing",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-repo/ai-film-analysis/issues",
        "Source": "https://github.com/your-repo/ai-film-analysis",
        "Documentation": "https://ai-film-analysis.readthedocs.io/",
    },
)