from py_paillier.py_paillier import *
from py_paillier.util import Euclid, calc_reduced_system_deductions
import random


if __name__ == '__main__':
    public_key: PaillierPublicKey = PaillierPublicKey(41 * 59)
    private_key: PaillierPrivateKey = PaillierPrivateKey(public_key, 41, 59)

    public_key.show_public_key()

    plaintext_1 = [random.randint(1, 41 * 59) for i in range(100)]
    encrypt_text_1 = public_key.encryption(plaintext_1)
    print(f"encrypt_text_1 = {encrypt_text_1}")

    # reduced_system_deductions = calc_reduced_system_deductions(public_key.n_square)
    # print(f"reduced_system_deductions = {reduced_system_deductions}")
    # print(f"encrypt_text_1 & reduced_system_deductions = {list(set(encrypt_text_1) & set(reduced_system_deductions))}")
    # for i in range(100):
    #     print(f"{Euclid.greatest_common_divisor(encrypt_text_1[i], public_key.n)}")

    # print(f"{2419 % public_key.n}")

    print(f"{private_key.decryption([2419])}")

    # plaintext_2 = [6, 7, 8, 9, 10]
    # encrypt_text_2 = public_key.encryption(plaintext_2)
    # print(f"encrypt_text_2 = {encrypt_text_2}")
    #
    # homomorphic = Homomorphic(public_key.n, public_key.g)
    #
    # multiple = homomorphic.raising_of_ciphertext_to_the_power_of_plaintext(encrypt_text_1, encrypt_text_2)
    # print(f"multiple = {multiple}")
    #
    # decrypt_multiple = private_key.decryption(multiple)
    # print(f"decrypt_multiple = {decrypt_multiple}")
    #
    # default_multiple = []
    # for i in range(5):
    #     default_multiple.append(
    #         (plaintext_1[i] * encrypt_text_2[i]) % public_key.n
    #     )
    #
    # print(f"default_multiple = {default_multiple}")

    # separation_texts = []
    # for i in range(5):
    #     separation_texts.append(decrypt_multiple[i] // plaintext_1[i])
    #
    # decrypt_separation = private_key.decryption(separation_texts)
    # print(f"decrypt_separation = {decrypt_separation}")
    #
    # double_decrypt = private_key.decryption(decrypt_multiple)
    # print(f"double_decrypt = {double_decrypt}")
    #
    # triple_decrypt = private_key.decryption(double_decrypt)
    # print(f"triple_decrypt = {triple_decrypt}")

    pass
