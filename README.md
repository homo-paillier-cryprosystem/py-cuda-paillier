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
___

Simple example:

```python
from py_paillier.py_paillier import PaillierKeyPairGenerator as KeyGen

# The standard key length in bits is 16
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

To use homomorphic properties, you need to import class `Homomorphic`:
```python
from py_paillier.py_paillier import PaillierKeyPairGenerator as KeyGen
from py_paillier.py_paillier import Homomorphic

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
