from setuptools import setup, find_packages
import os
setup(name="recursiveRouteChoice",
      package_dir={'': os.path.join('src', 'recursiveRouteChoice')},
      packages=find_packages(where=os.path.join('src', 'recursiveRouteChoice')))
