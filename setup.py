from pathlib import Path

import setuptools

curr_dir = Path(__file__).resolve().parent
with open(curr_dir / 'README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="diffabs",
    version="0.1",
    author="Xuankang Lin",
    author_email="xuankang.lin@gmail.com",
    description="Differentiable abstract domain implementations for neural network reasoning on PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/XuankangLin/DiffAbs",
    packages=setuptools.find_packages(),
    install_requires=['torch>=1.0.0'],
    tests_require=['pytest', 'torchvision'],
    classifiers=[
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)