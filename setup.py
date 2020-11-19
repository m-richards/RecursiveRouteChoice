from setuptools import setup, find_packages
import os

setup(name="recursiveRouteChoice",
      package_dir={'': os.path.join('src', )},
      packages=find_packages(where=os.path.join('src', )),
      version="0.1",
      description="Recursive logit model implementation in python",
      author="Matthew Richards",
      url="https://github.com/m-richards/RecursiveRouteChoice",
      classifiers=[
          "Programming Language :: Python :: 3",
      ],
      python_requires='>=3.6',
      install_requires=[
          "numpy",
          "scipy",
          "awkward1",
          "pandas"
      ]
      )
