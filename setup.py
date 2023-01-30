from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='schism_viz',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="holoviews visualizations for schism",
    license="MIT",
    author="Nicky Sandhu",
    author_email='psandhu@water.ca.gov',
    url='https://github.com/dwr-psandhu/schism_viz',
    packages=['schism_viz'],
    entry_points={
        'console_scripts': [
            'schism_viz=schism_viz.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='schism_viz',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
