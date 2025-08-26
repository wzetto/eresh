from setuptools import setup, find_packages

setup(
    name='eresh',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    license='Apache-2.0',
    description='Toolkit for high entropy diborides’ mechanical property derivation.',
    long_description=open('README.md').read(),
    install_requires=['numpy', 'matplotlib', 'pickle', 'scikit-learn', 'juliacall'],
    url='https://github.com/wzetto/eresh',
    author='Zhi Wang',
    author_email='wang.zhi.48u@st.kyoto-u.ac.jp',
)