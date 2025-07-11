from pathlib import Path
from setuptools import setup, find_packages


def read_requirements(path):
    return list(Path(path).read_text().splitlines())


reqs = read_requirements(Path('.').parent.joinpath("requirements.txt"))

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="LynseDB",
    version="0.2.0",
    description="A pure Python-implemented, lightweight, server-optional, multi-end compatible, vector database deployable locally or remotely.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Birch Kwok",
    author_email="birchkwok@gmail.com",
    url="https://github.com/BirchKwok/lynsedb",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=reqs,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12"
    ],
    license="Apache-2.0",
    keywords=["vector", "database", "vector database", "Similarity Search"],
    entry_points={
        'console_scripts': [
            'lynse=lynse.api.http_api.http_api.app:main',
        ],
    }
)
