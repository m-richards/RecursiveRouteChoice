from setuptools import setup, find_packages

setup(name="recursiveRouteChoice",
      version="0.1",
      description="Recursive logit model implementation in python",
      author="Matthew Richards",
      url="https://github.com/m-richards/RecursiveRouteChoice",
      package_dir={'': 'src'},
      packages=find_packages(where='src'),
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
