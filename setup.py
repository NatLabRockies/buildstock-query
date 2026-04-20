from setuptools import setup, find_packages

setup(
    name="buildstock_query",
    version="v0.2.0",
    description="Python library for querying and analyzing ResStock and ComStock",
    author="Rajendra Adhikari",
    packages=find_packages(),
    author_email="Rajendra.Adhikari@nrel.gov",
    package_dir={"buildstock_query": "buildstock_query"},
    package_data={"buildstock_query": ["**"]},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "pandas >= 2.0.0",
        "polars",
        "pyarrow >= 14.0.1",
        # "s3fs[boto3] >= 2022.8.2",
        "pyathena >= 3.19.0",
        "SQLAlchemy >= 2.0.0",
        "sqlalchemy2-stubs",
        "pandas-stubs",
        "inquirerpy >= 0.3.4",
        "types-PyYAML >= 6.0.12.2",
        "pydantic>=2",
        "PyYAML",
        "tabulate",
        "toml",
        "requests",
        "sqlglot",
        "filelock",
        "pyparsing"
    ],
    extras_require={
        "dev": [
            "pytest >= 7.1.3",
            "flake8 >= 5.0.4",
            "pdoc3 >= 0.10.0",
            "autopep8 >= 1.7.0",
            "dash-bootstrap-components >= 1.2.1",
            "dash-extensions >= 0.1.6",
            "dash-mantine-components == 0.10.2",
            "dash-iconify >= 0.1.2",
            "coverage >= 6.5.0",
            "plotly >= 5.10.0",
            "dash >= 2.6.2",
        ],
        "full": [
            "dash-bootstrap-components >= 1.2.1",
            "dash-extensions >= 0.1.6",
            "dash-mantine-components == 0.10.2",
            "dash-iconify >= 0.1.2",
            "plotly >= 5.10.0",
            "dash >= 2.6.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "upgrades_analyzer=buildstock_query.tools.upgrades_analyzer:main",
            "upgrades_visualizer=buildstock_query.tools.upgrades_visualizer:main",
        ]
    },
)
