import os
import re

import setuptools

here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, "src", "rubric_eval", "__init__.py")) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find `__version__`.")

PACKAGES_DEV = [
    "pre-commit>=3.2.0",
    "black>=23.1.0",
    "isort",
    "pytest",
    "pytest-mock",
    "pytest-skip-slow",
    "pytest-env",
    "python-dotenv",
]
PACKAGES_ANALYSIS = ["seaborn", "matplotlib", "jupyterlab"]
PACKAGES_ALL = PACKAGES_ANALYSIS + PACKAGES_DEV

setuptools.setup(
    name="rubric_eval",
    version=version,
    description="RubricEval: Scalable Expert Evaluation of Language Models",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    author="TODO dream team",
    install_requires=[
        "python-dotenv",
        "datasets",
        "pandas",
        "huggingface_hub",
        "fire",
        "alpaca_eval>=0.6.5",
    ],
    extras_require={
        "analysis": PACKAGES_ANALYSIS,
        "dev": PACKAGES_DEV,
        "all": PACKAGES_ALL,
    },
    python_requires=">=3.10",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "rubric_eval=rubric_eval.main:main",
        ],
    },
    include_package_data=True,
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)
