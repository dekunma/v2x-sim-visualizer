import pathlib
from distutils.core import setup
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="v2x-sim-visualizer",
    version="0.0.1",
    package_data={
        "": ["*.so"],
    },
    packages=setuptools.find_packages(),
    license="apache-2.0",  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description="A library for collaborative perception.",
    author="AI4CE Lab @NYU",
    author_email="dm4524@nyu.edu",
    url="https://ai4ce.github.io/V2X-Sim/",
    download_url="",
    keywords=[
        "computer-vision",
        "deep-learning",
        "autonomous-driving",
        "collaborative-learning",
        "knowledge-distillation",
        "communication-networks",
        "multi-agent-learning",
        "multi-agent-system",
        "3d-object-detection",
        "graph-learning",
        "point-cloud-processing",
        "v2x-communication",
        "multi-agent-perception",
        "3d-scene-understanding",
    ],  # Keywords that define your package best
    install_requires=[
        "matplotlib",
        "numpy",
        "nuscenes-devkit",
        "pyquaternion",
        "Pillow",
        "setuptools",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",  # Specify which pyhton versions that you want to support
    ],
)