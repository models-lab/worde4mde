import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

#with open("requirements.txt", "r", encoding="utf-8") as fh:
#    requirements_contents = fh.read().split("\n")

setuptools.setup(
    name="worde4mde",
    version="1.1",
    author="Jesús Sánchez Cuadrado",
    author_email="jesusc@um.es",
    description="Embeddings for software modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/models-lab/worde4mde",
    project_urls={
        "Bug Tracker": "https://github.com/models-lab/worde4mde/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.0",
#    install_requires=requirements_contents
)
