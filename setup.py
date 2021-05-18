from setuptools import find_packages, setup
from lendingclub_scoring import __version__

setup(
    name='lendingclub_scoring',
    packages=find_packages(exclude=['tests', 'tests.*']),
    setup_requires=['wheel'],
    version=__version__,
    description='LindingClub Scoring Demo powered by DBX',
    author=''
)
