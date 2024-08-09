from distutils.core import setup

setup(
  name = 'autotda',         # How you named your package folder (MyLib)
  packages = ['autotda'],   # Chose the same as "name"
  version = '0.0.8',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Human in the loop Data Augmentation',   # Give a short description about your library
  author = 'Zeger Mouw',                   # Type in your name
  author_email = 'z.f.mouw@student.tudelft.nl',      # Type in your E-Mail
  url = 'https://github.com/delftdata/hci-auto-feat',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/delftdata/hci-auto-feat/archive/refs/tags/0.0.1.tar.gz',    # I explain this later on
  keywords = ['SOME', 'MEANINGFULL', 'KEYWORDS'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'autogluon',
        "deprecation",
        "dsplot",
        "graphviz",
        "ITMO-FS",
        "joblib",
        "matplotlib", 
        "neo4j",
        "networkx",
        "notebook",
        "numpy",
        "pandas",
        "pydot",
        "pyvis",
        "qpsolvers[daqp]",
        "scikit-learn",
        "seaborn",
        "tqdm",
        "typer",
        "valentine == 0.1.6"
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.10',      #Specify which pyhton versions that you want to support
  ],
)