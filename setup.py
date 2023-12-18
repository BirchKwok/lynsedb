from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(path):
    return list(Path(path).read_text().splitlines())


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='MinVectorDB',
    version="0.0.4",
    description='MinVectorDB is a simple vector storage and query database implementation, ' \
                'providing clear and concise Python APIs aimed at lowering the barrier to using vector databases.',
    keywords='vector database',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires=">=3.9",
    url='https://github.com/BirchKwok/MinVectorDB',
    author='Birch Kwok',
    author_email='birchkwok@gmail.com',
    install_requires=[
        'numpy>=1.17.0',
        'spinesUtils>=0.3.13'
    ],
    zip_safe=False,
)
