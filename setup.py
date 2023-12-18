from setuptools import setup, find_packages

setup(
    name="franka-valve",
    version="0.0.1",
    description="Rotate a valve using Franka Panda with Reinforcement Learning in mujoco simulation."
    author="twkang43",
    author_email="twkang43@gmail.com",
    url="https://github.com/twkang43/franka_simulation/tree/feat/setting-env/py_src",
    install_requires=["gym", "numpy", "gymnasium", "tensorborad", "torchvision", "torchaudio", "pytorch-cuda", "pyparsing", "stable-baselines3", "sb3-contrib", "six"],
    packages=find_packages(exclude=[]),
    python_requires=">=3.8",
    package_data={},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)