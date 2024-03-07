from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(path):
    return list(Path(path).read_text().splitlines())


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='MinVectorDB',
    version="0.2.3",
    description='MinVectorDB is a pure Python-implemented, lightweight, stateless vector, locally deployed database' \
                'that offers clear and concise Python APIs, aimed at lowering the barrier to ' \
                'the use of vector databases.',
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
        'spinesUtils>=0.3.13',
        'torch>=2.0.0',
        'msgpack>=1.0.2',
        'h5py>=3.4.0',
        'scikit-learn>=1.0.0',
        'cloudpickle>=2.0.0',
        'numexpr>=2.7.3',
        'pyroaring>=0.4.5',
    ],
    zip_safe=False,
)
