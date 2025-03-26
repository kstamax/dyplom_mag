from setuptools import setup, find_packages

setup(
    name="dyplom_mag",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of your library",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kstamax/dyplom_mag",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "ultralytics"
        # List your package dependencies here, e.g. "numpy", "requests"
    ],
)
