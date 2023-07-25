import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

version = 1.0

setuptools.setup(
    name="accomatic",
    version=version,
    author="Hannah Macdonell",
    author_email="hannah.macdonell@carleton.ca",
    description="Running analysis on point-scale model simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hgmacc/accomatic-web",
    packages=['accomatic'],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Atmospheric Science"
        ],
    entry_points={
        'console_scripts': [
            'acco = accomatic:acco',
        ]},

    install_requires=['xarray',
                      'tomlkit',
                      'numpy',
                      'pandas',
                      'netCDF4',
                      'scipy',
                      'scikit-learn',
                      'pandas',
                      'typing',
                      'datetime',
                      'matplotlib',
                      'regex',
                      'tomlkit']
                      
)