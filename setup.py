import setuptools

setuptools.setup(
    name='cut_predictor',
    version='0.0.1',
    author='Aida Farahani, Payam Atoofi, Julien Vitay',
    author_email='julien.vitay@informatik.tu-chemnitz.de ',
    description='AutoML class to predict deviations from 1D positions.',
    platforms='Posix; MacOS X; Windows',
    packages=setuptools.find_packages(where='./src'),
    package_dir={
        '': 'src'
    },
    include_package_data=True,
    install_requires=(
        'numpy',
    ),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)