from setuptools import setup, find_packages

setup(
    name="voxelsss",
    version="0.1.0",
    description="Voxel-based structure simulation solvers",
    author="Simon Daubner",
    author_email="s.daubner@imperial.ac.uk",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyvista",
        "matplotlib",
        "torch"
    ],
    license="MIT license",
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Environment :: GPU',
        'Environment :: GPU :: NVIDIA CUDA',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    zip_safe=False,
)