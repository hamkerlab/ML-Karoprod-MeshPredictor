import setuptools

setuptools.setup(
    name='mesh_predictor',
    version='1.0.0',
    author='Aida Farahani, Payam Atoofi, Julien Vitay',
    author_email='julien.vitay@informatik.tu-chemnitz.de ',
    description='AutoML predictors to predict the outcome of FEM simulations.',
    platforms='Posix; MacOS X',
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