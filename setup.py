from setuptools import setup

setup(name='radar_odometry',
      version='0.1',
      description='A simple odometry estimation package using radar',
      url='https://github.com/hflemmen/radar_odometry',
      author='Henrik',
      author_email='henrik.d.flemmen@ntnu.no',
      license='MIT',
      packages=['radar_odometry'],
      install_requires=[
          'opencv-python',
          'numpy',
          'polarTransform',
          'matplotlib',
          'gtsam',
          'manifpy',
          'pymap3d',
          # 'numba',
      ],
      zip_safe=False)
