from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(
    name="py_cuda_paillier",
    version="0.0.3",
    author="Vsevolod Vlasov",
    author_email="cold.vv.ss@gmail.com",
    description="Adapted Paillier Cryptosystem to CUDA architecture",
    long_description=readme,
    long_description_content_type='text/markdown',
    url="https://github.com/homo-paillier-cryprosystem/py-cuda-paillier/",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
