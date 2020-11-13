import setuptools
import codecs

with codecs.open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="parot",
    version="1.0.0",
    author="FiveAI Ltd",
    author_email="edward.ayers@five.ai",
    description="Tool for training and testing neural networks for " +
            "robustness and verified robustness.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    install_requires=[
        "tensorflow==2.3.1",
        "tensorflow-gpu==1.14.0",
        "tensorboard==1.14.0",
        "matplotlib>=3.1.3",
        "numpy==1.16.0",
        "tqdm>=4.42.1"
    ],
    include_package_data=True
)
