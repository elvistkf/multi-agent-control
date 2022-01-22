from setuptools import setup
setup(
    name='multiagentcontrol',
    version='0.1',
    description='A simulation package for multi-agent control systems',
    url='#',
    author='Elvis Tsang',
    author_email='elviskftsang@gmail.com',
    license='MIT',
    packages=['multiagentcontrol'],
    install_requires=[
        '<matplotlib>',
        '<numpy>',
        '<scipy>',
        '<networkx>',
        '<pathos>'
    ],
    zip_safe=False
)