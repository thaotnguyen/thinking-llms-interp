from setuptools import find_packages, setup

setup(
    name="deepseek_steering",
    packages=find_packages(where="."),
    package_dir={"": "."},
    package_data={
        "deepseek_steering": ["data/**/*"],
    },
    include_package_data=True,
)