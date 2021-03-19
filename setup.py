import os

from setuptools import find_packages, setup

description = "ML to analyze voting decisions from FiveThirtyEight polling data."

try:
    this_path = os.path.dirname(os.path.abspath(__file__))
    fn_readme = os.path.join(this_path, "README.md")
    with open(fn_readme) as fh:
        long_description = fh.read()
except OSError:
    long_description = description

setup(
    name="voting_ml",
    version="0.0.0",
    packages=find_packages(),
    zip_safe=True,
    author="Tommy Waltmann, Sikandar Hanif, Prachi Atmasiddha, and Sergio Garcia",
    author_email="tomwalt@umich.edu",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
)
