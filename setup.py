from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(path):
    return list(Path(path).read_text().splitlines())


reqs = read_requirements(Path('.').parent.joinpath("requirements.txt"))

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='LynseDB',
    version="0.1.5",
    description='A pure Python-implemented, lightweight, server-optional, '
                'multi-end compatible, vector database deployable locally or remotely.',
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
    install_requires=reqs,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'lynse=lynse.api.http_api.http_api.app:main',
        ],
    },
)
