from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='deepreplay',
      version='0.1.0a4',
      install_requires=['matplotlib', 'numpy', 'h5py', 'seaborn', 'keras'],
      description='"Hyper-parameters in Action!" visualizing tool for Keras models.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/dvgodoy/deepreplay',
      author='Daniel Voigt Godoy',
      author_email='datagnosis@gmail.com',
      keywords=['keras', 'hyper-parameters', 'animation', 'plot', 'chart'],
      license='MIT',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Visualization',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3'
      ],
      packages=find_packages(),
      zip_safe=False)
