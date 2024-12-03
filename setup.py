from setuptools import setup

setup(
    name='SpacePhyML',
    version='0.1.0',
    description='A Framework for distributin Machine Learning Datasets for Space and Plasma physics.',
    url='https://github.com/Jonah-E/SpacePhyML',
    author='Jonah Ekelund',
    author_email='jonahek@kth.se',
    license='MIT License',
    packages=['spacephyml'],
    install_requires=['torch',
                      'pandas',
                      'cdflib',
                      ],

    classifiers=[
    ],
)

