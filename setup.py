from setuptools import setup, find_packages

setup(name='maddpg',
      version='0.0.1',
      description='Multi-Agent Deep Deterministic Policy Gradient',
      url='https://github.com/openai/maddpg',
      author='Igor Mordatch',
      author_email='mordatch@openai.com',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)
