from setuptools import setup, find_packages

setup(
    name="xdecisiontree",
    version="1.0.0",
    author="Reijo Jaakkola",
    author_email="jaakkolareijo@hotmail.com",
    description="Decision tree classifier with human-readable rule extraction",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ReijoJaakkola/xdecisiontree",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)