from setuptools import setup, find_packages
from fugue_incubator_version import __version__
import os

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()


def get_version() -> str:
    tag = os.environ.get("RELEASE_TAG", "")
    if "dev" in tag.split(".")[-1]:
        return tag
    if tag != "":
        assert tag == __version__, "release tag and version mismatch"
    return __version__


setup(
    name="fugue-incubator",
    version=get_version(),
    packages=find_packages(),
    description="Fugue based experimental projects",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    author="Han Wang",
    author_email="goodwanghan@gmail.com",
    keywords="fugue incubator experiment",
    url="http://github.com/fugue-project/fugue-incubator",
    install_requires=["fugue>=0.5.0", "scikit-learn", "matplotlib"],
    extras_require={
        "hyperopt": ["hyperopt"],
        "notebook": ["notebook", "jupyterlab"],
        "all": ["hyperopt", "notebook", "jupyterlab"],
    },
    classifiers=[
        # "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires=">=3.6",
)
