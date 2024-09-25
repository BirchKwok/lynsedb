<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/BirchKwok/LynseDB/blob/main/logo/logo.png">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/BirchKwok/LynseDB/blob/main/logo/logo.png">
    <img alt="LynseDB logo" src="https://github.com/BirchKwok/LynseDB/blob/main/logo/logo.png" height="100">
  </picture>
</div>
<br>

<p align="center">
  <a href="https://discord.com/invite/rcYK5nYF"><img src="https://img.shields.io/badge/Discord-Online-brightgreen" alt="Discord"></a>
  <a href="https://badge.fury.io/py/LynseDB"><img src="https://badge.fury.io/py/LynseDB.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/LynseDB/"><img src="https://img.shields.io/pypi/pyversions/LynseDB" alt="PyPI - Python Version"></a>
  <a href="https://github.com/BirchKwok/LynseDB/actions/workflows/python-tests.yml"><img src="https://github.com/BirchKwok/LynseDB/actions/workflows/python-tests.yml/badge.svg" alt="Python testing"></a>
  <a href="https://github.com/BirchKwok/LynseDB/actions/workflows/docker-tests.yml"><img src="https://github.com/BirchKwok/LynseDB/actions/workflows/docker-tests.yml/badge.svg" alt="Docker build"></a>
</p>


**LynseDB** is a vector database implemented purely in Python, designed to be lightweight, server-optional, and easy to deploy locally or remotely. It offers straightforward and clear Python APIs, aiming to lower the entry barrier for using vector databases.

It focuses on achieving 100% recall, prioritizing recall accuracy over high-speed search performance. This approach ensures that users can reliably retrieve all relevant vector data, making LynseDB particularly suitable for applications that require responses within hundreds of milliseconds.

## LynseDB features

⚡ **Server-optional, simple parameters, simple API.**

⚡ **Fast, memory-efficient, easily scales to millions of vectors.**

⚡ **Based on a generic Python software stack, platform-independent, highly versatile.**

⚡ **Recall-prioritized design, lifecycle search caching technology, FieldExpression fast filtering, Field multi-type indexing, and other user-centric features**

## Some Defects You Should Know

- Not yet backward compatible

LynseDB is actively being updated, and API backward compatibility is not guaranteed. You should use version numbers as a strong constraint during deployment to avoid unnecessary feature conflicts and errors.

- Data size constraints

Although our goal is to enable brute force search or inverted indexing on billion-scale vectors, we currently still recommend using it on a scale of millions of vectors or less for the best experience.

- Python's native api is not process-safe

The Python native API is recommended for use in single-process environments, whether single-threaded or multi-threaded; for ensuring process safety in multi-process environments, please use the HTTP API.


## Installation

### Prerequisite

- python version >= 3.9
- Owns one of the operating systems: Windows, macOS, or Ubuntu (or other Linux distributions). The recommendation is for the latest version of the system, but non-latest versions should also be installable, although they have not been tested.
- Memory >= 4GB, Free Disk >= 4GB.

### Install Client API package (Mandatory)

```shell
pip install LynseDB
```

### If you wish to use Docker (Optional)

You must first [install Docker](https://docs.docker.com/engine/install/) on the host machine.

After installing the [Client API package](#install-client-api-package-mandatory):

```shell
docker pull birchkwok/lynsedb:latest
```
