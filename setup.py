from setuptools import setup, find_packages

setup(
    name="fastaug",
    version="20201121",
    keywords=["fastaug", ],
    description="eds sdk",
    long_description="Fastaug is an NLP library for data augmentation with high speed.",
    license="WTFPL Licence",

    url="https://github.com/dugu9sword/fastaug",
    author="dugu9sword",
    author_email="dugu9sword@163.com",

    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=[
        "nltk",
        "numpy",
        "importlib_resources"
    ],
    zip_safe=False,

    scripts=[],
    entry_points={}
)