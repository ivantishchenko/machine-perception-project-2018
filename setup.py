"""Setup module for project."""

from setuptools import setup, find_packages

setup(
        name='mp18-project-skeleton',
        version='0.1',
        description='Code for Machine Perception project Hand Joint Recognition',

        author='Ivan Tishchenko, Mickey Vänskä',
        author_email='tivan@student.ethz.ch, mickeyv@student.ethz.ch',

        packages=find_packages(exclude=[]),
        python_requires='>=3.5',
        install_requires=[
            'coloredlogs',
            'h5py',
            'numpy',
            'opencv-python',
            'pandas',
            'tensorflow',
            'git+https://github.com/aleju/imgaug'
        ],
)