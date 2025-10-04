from setuptools import setup, find_packages

setup(
    name="lit-lm",
    version="0.1.0",
    description="Lightning-based training framework for causal language models",
    author="Marc harary",
    packages=find_packages(exclude=("tests", "configs", "scripts")),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.1.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "wandb",
        "omegaconf",
        "hydra-core",
    ],
    entry_points={
        "console_scripts": [
            "litlm-train=scripts.train:main",
        ],
    },
)
