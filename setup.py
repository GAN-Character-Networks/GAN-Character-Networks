import setuptools
from distutils.core import setup

with open("README.md", "r") as f:
    long_description = f.read()

author_emails = [
    "adel.moumen@univ-avignon.fr",
    "nicolas.bataille@univ-avignon.fr",
    "gabriel.desbouis@univ-avignon.fr",
]

setup(
    name="vroom",
    version="0",
    description="Automatic extraction library of character networks.",
    package_dir={"": "src"},
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Adel Moumen & Nicolas Bataille & Gabriel Desbouis",
    author_email=author_emails,
    classifiers=[
        "License :: OSI Approved :: Apache 2.0 License",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    install_requires=["torch", "transformers", "jinja2", "openai"],
    extras_require={"dev": ["pytest>=7.0", "twine>=4.0.2"]},
    python_requires=">=3.6",
)
