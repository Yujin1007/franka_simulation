import sys
from setuptools import setup, find_packages

setup(
    name="franka-valve",
    version="0.0.0",
    description="Rotate a valve using Franka Research 3 with Reinforcement Learning in mujoco simulation.",
    author="Yujin1007, twkang43",
    author_email="yujin1004k@gmail.com, twkang43@gmail.com",
    url="https://github.com/Yujin1007/franka_simulation",
    license="MIT",
    keywords=["reinforcement learning", "RL", "robotics", "robot", "franka", "emika", "panda", "fr3"],
    platforms=[sys.platform],

    install_requires=[
                        "setuptools==65.5.0",
                        "pip==21",
                        "wheel==0.38.0",
                        "numpy>=1.24.1", 
                        "torch>=2.1.1",
                        "torchvision>=0.16.1",
                        "torchaudio>=2.1.1",
                        "mujoco==2.3.5",
                        "gym==0.19.0", 
                        "gymnasium==0.28.1",
                        "pyparsing>=3.0.9",
                        "scipy>=1.8.1",
                        "six>=1.16.0"
                     ],
    packages=find_packages(include=["franka_valve", 
                                    "franka_valve.fr3_envs", 
                                    "franka_valve.fr3_envs.bases", 
                                    "franka_valve.models.classifier", 
                                    "franka_valve.models.tqc", 
                                    "franka_valve.utils"
                                    ]),
    python_requires=">=3.8",

    package_data={
        "franka_valve.assets": ["*"],
        "franka_valve.models.classifier": ["model_cclk.pt", "model_clk.pt"],
        "franka_valve.models.tqc.model.default_model": ["*"],
        "franka_valve.fr3_envs.jsons.candidates": ["*"],
        "franka_valve.fr3_envs.jsons.contacts": ["contact.json"],
    },
    include_package_data=True
)
