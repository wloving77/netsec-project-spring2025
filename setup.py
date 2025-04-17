# setup.py
import setuptools

setuptools.setup(
    name="gan_ids_project",
    version="0.1.0",
    author="Anders Gyllenhoff",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wloving77/netsec-project-spring2025",
    packages=["src"],
    package_dir={"src": "gan_ids_project/src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "jupyter",
        "scapy",
        "sdv",
        "torch",
        "gdown",
        "tensorflow",
    ],
)