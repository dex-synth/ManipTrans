import pathlib
import os

from setuptools import setup, find_packages

PKG_NAME = "maniptrans"
VERSION = "0.0.1"
EXTRAS = {}
H2R = os.environ.get("MANIPTRANS_PROFILE", "native") == "h2r"


def _read_file(fname):
    with pathlib.Path(fname).open() as fp:
        return fp.read()


def _read_install_requires():
    filename = "requirements-h2r.txt" if H2R else "requirements.txt"
    return [line.strip() for line in pathlib.Path(filename).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


setup(
    name=PKG_NAME,
    version=VERSION,
    author="MANIPTRANS Developers",
    description="research project",
    long_description=_read_file("README.md"),
    long_description_content_type="text/markdown",
    keywords=["Robotics", "Reinforcement Learning", "Machine Learning"],
    license="Apache License, Version 2.0",
    packages=find_packages(include=["lib", "lib.*", "maniptrans_h2r", "maniptrans_h2r.*"])
             if H2R else find_packages(include=f"{PKG_NAME}.*"),
    include_package_data=True,
    zip_safe=False,
    entry_points={"console_scripts": []},
    install_requires=_read_install_requires(),
    python_requires="==3.11.*" if H2R else "==3.8.*",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Robotics",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
    ],
)
