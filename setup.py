from setuptools import setup, find_packages
from src.deep_search_lightning import __version__

setup(
    version=__version__,
    name="deep_search_lightning",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests",
        "python-dotenv",
        "duckduckgo_search",
        "openai",
        "asyncio",
        "baidusearch",
        "httpx",
        "beautifulsoup4",
        "streamlit",
    ],
    python_requires=">=3.11",
description="A powerful web search toolkit with multi-engine support",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="positive666",  
    author_email="286040359@qq.com",
    url="https://github.com/positive666/deep_search_lightning",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
