from setuptools import setup, find_packages

setup(
    name="mlops_heart_disease",
    version="0.1.0",
    description="End-to-end MLOps pipeline for Heart Disease prediction",
    author="Lokesh B",
    packages=["src", "app"],
    python_requires=">=3.9",
)