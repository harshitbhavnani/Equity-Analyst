from setuptools import setup

with open('./README.rst') as f:
    readme = f.read()

setup(
    name='Equity Analyst',
    version='1.0.2',
    packages=['equityanalyst'],
    url='https://github.com/harshitbhavnani/Equity-Analyst',
    license='MIT',
    author='Harshit Bhavnani',
    author_email='harshit.bhavnani@gmail.com',
    description='Equity Analyst is an Open-Source Python package with the key function of accepting the ticker name of the stock and predicting its future value based on historical data.',
    long_description=readme

)
