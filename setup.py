from setuptools import find_packages, setup

setup(
    name="python-polaris",
    version="0.1.1",
    packages=find_packages(where="polaris"),
    install_requires=[
        # Add any dependencies here, e.g., requests
    ],
)
