from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='schismviz',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="holoviews visualizations for schism",
    license="MIT",
    author="Nicky Sandhu",
    author_email='psandhu@water.ca.gov',
    url='https://github.com/cadwrdeltamodeling/schismviz',
    packages=['schismviz'],
    entry_points={
        'console_scripts': [
            'schismviz=schismviz.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='schismviz',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
