from setuptools import setup

setup(
    name='keras_video_classifier',
    packages=['keras_video_classifier'],
    include_package_data=True,
    install_requires=[
        'flask',
        'keras',
        'tensorflow',
        'numpy',
        'matplotlib',
        'opencv-python',
        'h5py',
        'scikit-learn'
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)