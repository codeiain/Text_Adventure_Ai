"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="text_adventure_ai",  # Required
    version="0.0.2",  # Required
    description="Ais for Text Adventures",  # Optional
    # This should be your name or the name of the organization which owns the
    # project.
    author="Code Iain",  # Optional
    # This should be a valid email address corresponding to the author listed
    # above.
    author_email="codeiain@outlook.com",  # Optional
    keywords="sample, setuptools, development",  # Optional
    packages=find_packages(),
    python_requires=">=3.7, <4",
    install_requires=["spacy", "rich", "python-dotenv", "git+https://github.com/codeiain/FeatureFlags.git"],  # Optional
    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install sampleproject[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    # extras_require={  # Optional
    #     "dev": ["check-manifest"],
    #     "test": ["coverage"],
    # },
    # If there are data files included in your packages that need to be
    # installed, specify them here.pip install
    # package_data={  # Optional
    #     "sample": ["package_data.dat"],
    # },
    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/distutils/setupscript.html#installing-additional-files
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[("my_data", ["data/data_file"])],  # Optional
    include_package_data=True,
    package_data={'': ['ner/training_data/*.json']},
    project_urls={  # Optional
        "Bug Reports": "https://github.com/pypa/sampleproject/issues",
        "Funding": "https://donate.pypi.org",
        "Say Thanks!": "http://saythanks.io/to/example",
        "Source": "https://github.com/pypa/sampleproject/",
    },
)