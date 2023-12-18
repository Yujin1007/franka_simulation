from setuptools import setup, find_packages

setup(
    name="franka-valve",
    version="0.0.0.1",
    description="Rotate a valve using Franka Panda with Reinforcement Learning in mujoco simulation.",
    author="twkang43",
    author_email="twkang43@gmail.com",
    url="https://github.com/Yujin1007/franka_simulation",
    license="MIT",
    keywords=["reinforcement learning", "RL", "franka", "emika", "panda"],

    install_requires=["gym", "numpy", "gymnasium", "tensorborad", "torchvision", "torchaudio", "pytorch-cuda", "pyparsing", "stable-baselines3", "sb3-contrib", "six"],
    packages=find_packages(where="py_src"),
    package_dir={"":"py_src"},
    python_requires=">=3.8",

    package_data={},
    classifiers=[
        "Programming Language::Python::3",
        "License :: OSI Approved :: MIT License",
        "Operating System::Linux"
    ]
)