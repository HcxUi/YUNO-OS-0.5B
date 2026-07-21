from setuptools import setup, find_packages

setup(
    name="yuno-llm",
    version="0.5.0",
    description="YUNO-LLM Personal AI Operating System",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="YUNO-LLM Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "yuno=yuno_llm.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
