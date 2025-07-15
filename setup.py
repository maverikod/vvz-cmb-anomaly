from setuptools import setup, find_packages

setup(
    name="cmb-anomaly",
    version="0.1.0",
    description="Statistical analysis and anomaly detection on Planck CMB maps (SMICA 2018)",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Vasily VZ",
    author_email="your@email.com",
    url="https://github.com/yourusername/cmb-anomaly",
    packages=find_packages(),
    py_modules=["main", "convert_fits"],
    install_requires=[
        "numpy>=1.20",
        "astropy>=5.0"
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "cmb-anomaly=main:main",
            "convert-fits=convert_fits:main"
        ]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    license="MIT",
) 