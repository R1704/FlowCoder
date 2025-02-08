from setuptools import setup, find_packages

setup(
    name="flowcoder",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchgfn @ git+https://github.com/GFNOrg/torchgfn.git"
    ],
    python_requires=">=3.10",
)