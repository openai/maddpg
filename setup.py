from setuptools import setup, find_packages

setup(name='multiagent_rl',
      version='0.0.1',
      description='Multi-Agent Reinforcement Learning Algorithms',
      url='https://github.com/openai/multiagent_rl',
      author='Igor Mordatch',
      author_email='mordatch@openai.com',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)
