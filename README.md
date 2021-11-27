# py-cuda-paillier
A Python 3 library implementing the adapted Paillier Partially Homomorphic Encryption to CUDA architecture.

The homomorphic properties of the paillier crypto system are:
+ encrypted numbers can be added together:
    + encrypted numbers and non-encrypted numbers;
    + encrypted numbers and encrypted numbers;
+ Encrypted numbers can be multiplied.

# Annotation
You can translate your data through any encoding you like, and then use this library for encryption.

# Usage
To use this library, install:
+ CUDA Toolkit v11.5 and higher;
+ Anaconda and numba;
+ you may also need Microsoft Visual C ++ Builder.

Use the official sites and this video as a tutorial as a guide for installation:

[![Watch video](https://i.ytimg.com/vi_webp/vMZ7tK-RYYc/maxresdefault.webp)](https://www.youtube.com/watch?v=vMZ7tK-RYYc)

The use of this library is limited by the size of the plain text, at the moment the maximum character value that can be encrypted is "9796" according to the Unicode table.
This library is an extended version of the py_paillier library [py-paillier][https://test.pypi.org/project/py-paillier].

This library includes some changes:
+ standard key size in bits;
+ encryption and decryption functions adapted to the CUDA architecture;
+ adapted homomorphic properties to the CUDA architecture.
___

**Simple example**:

```python
from py_cuda_paillier.py_cuda_paillier import PaillierKeyPairGenerator as KeyGen

# The standard key length in bits is 12
public_key, private_key = KeyGen().paillier_key_pair_generation()

public_key.show_public_key()
private_key.show_private_key()

# Your plaintext as a list of numbers from the encoding 
plaintext = [1, 2, 3, 4, 5]
encrypt_text = public_key.encryption(plaintext)
print(f"{encrypt_text}")

decrypt_text = private_key.decryption(encrypt_text)
print(f"{decrypt_text}")
```

**To use homomorphic properties, you need to import class** `Homomorphic`:

```python
from py_cuda_paillier.py_cuda_paillier import PaillierKeyPairGenerator as KeyGen
from py_cuda_paillier.py_cuda_paillier import Homomorphic

# The changed key length in bits is 128
public_key, private_key = KeyGen().paillier_key_pair_generation(128)
homomorphic = Homomorphic(public_key.n, public_key.g)

plaintext_1 = [20094, 25774, 16518, 18209, 22329]
encrypt_text_1 = public_key.encryption(plaintext_1)

plaintext_2 = [19025, 32145, 17900, 29522, 30085]
encrypt_text_2 = public_key.encryption(plaintext_2)

addition_two_ciphertexts = homomorphic.addition_of_two_ciphertexts(
    encrypt_text_1,
    encrypt_text_2
)
```
---

**Extended usage GPU**:

_All GPU functions start with "cuda\_" in their respective classes._

```python
from py_cuda_paillier.py_cuda_paillier import PaillierPublicKey, PaillierPrivateKey, CudaConfig
from py_cuda_paillier.py_cuda_paillier import PaillierKeyPairGenerator as pkpg
import random

MAX_POWER = 12

public_key: PaillierPublicKey
private_key: PaillierPrivateKey
public_key, private_key, p, q = pkpg().paillier_key_pair_generation(bit_key_length=MAX_POWER, return_pq=True)

cuda_config = CudaConfig(
    threads_per_block=100,
    blocks=200
)

plaintext = [
    random.randrange(100, public_key.n) for _ in range(cuda_config.threads_per_block * cuda_config.blocks)
]

gpu_enc_text = public_key.cuda_encryption(
    plaintext_as_digits_list=plaintext,
    cuda_config=cuda_config,
    don_t_use_r=False
)
dec_text_from_gpu = private_key.cuda_decryption(gpu_enc_text, cuda_config)
```

# Notes:

The clear text to be encrypted, decrypted or used in homomorphic properties MUST have a number of characters equal to the result of multiplying the parameters of the CudaConfig class, that is:

```python
from py_cuda_paillier.py_cuda_paillier import CudaConfig

cuda_config = CudaConfig(
    threads_per_block=100,
    blocks=200
)

plaintext = [
    _ for _ in range(cuda_config.threads_per_block * cuda_config.blocks)
]
if len(plaintext) != (cuda_config.threads_per_block * cuda_config.blocks):
    print("ERROR")
else:
    pass

```


[https://test.pypi.org/project/py-paillier]: https://test.pypi.org/project/py-paillier