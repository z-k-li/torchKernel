import setuptools
#from . import __author__, __version__

setuptools.setup(
    name="torchKernel",
    version='1.0.0',
    description="",
    author='Daniel Deidda',
    author_email='danieldeidda@npl.co.uk',
    license='Apache-2.0',
    packages=["torchKernel",
              "torchKernel.notebooks",
              "torchKernel.utils",
              "torchKernel.algorithms",
              "torchKernel.architectures",
              "torchKernel.kernel",
              ],
              
    entry_points={
                  "console_scripts": [
                                        "neuralKEM=artcertainty.algorithms.neuralKEM:main"
                                     ],
                 },
    python_requires='>=3.10',
    install_requires=['numpy>=1.21', 
                      'torch>=1.13.0',
                      'tqdm>=4.66.4',
                       'brainweb>=1.6.2',
                       'deprecation>=2.1.0',
                       'pytorch-ignite>=0.5.1',
                       'seaborn>=0.13.2',
                       'pandas>=2.2.3',
                       'matplotlib>=3.10',
                       'type-docopt>= 0.8.2'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        #'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux', 
        'Programming Language :: Python :: 3.10',
    ],
    #cmdclass={"test": Run_TestSuite, "tox": ToxTest},
)
