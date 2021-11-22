"""Paillier encryption library for partially homomorphic encryption."""

from py_paillier.util import Euclid, PrimeDigit, check_plaintext

DEFAULT_BIT_KEY_LENGTH = 16


def l_func(u: int, _n: int):
    """ Helper function L takes an integer number of occurrences (u - 1) in n.
    Examples:
        13 / 4 = 3 * (1 / 4) -> result will be 3                  \n
        25 / 5 = 5 * (0 / 5) -> result will be 5                  \n
        67 / 11 = 6 * (1 / 11) -> result will be 6                \n
    :param u: int as dividend
    :param _n: int as divisor
    :return: int as quotient
    """
    return (u - 1) // _n


class PaillierPublicKey(object):
    """Contains a public key and associated encryption method.

    Args:
        :arg n (int): part of public key - see [1] \n
        :arg g (int): part of public key - see [1] \n
        :arg n_square (int): (n ** 2), stored for calculations \n

    Links:
        [1] - https://en.wikipedia.org/wiki/Paillier_cryptosystem#Key_generation
    """

    def __init__(self, n):
        # public key
        self.n = n
        self.g = self.generation_g(self.n)

        # parameters for calculations
        self.n_square = n ** 2

    @staticmethod
    def generation_g(_n: int):
        """Function for generating g as part of a public key.

        :param _n: int as modulo
        :return: g (int) as large prime
        """
        g = PrimeDigit().generating_a_large_prime_modulo(_n ** 2)
        return g

    def show_public_key(self):
        """Public key display function

        :return: None
        """
        print(f"public_key: {self.n}, {self.g}")

    def encryption(self, plaintext_as_digits_list: [int], don_t_use_r: bool = False):
        """Encryption function of plain text presented as a list of unencrypted numbers.

        :param don_t_use_r: (bool) optional, used for homomorphic encryption function
        :param plaintext_as_digits_list: (list[int]) - list of unencrypted numbers
        :return: empty list [] or non-empty list including encrypted digits
        """
        plaintext_is_current: bool = check_plaintext(
            plaintext_as_digits_list, self.n
        )
        if plaintext_is_current:
            encrypt_text_as_digits_list = []
            if don_t_use_r:
                for digit in plaintext_as_digits_list:
                    encrypt_text_as_digits_list.append(
                        ((self.g ** digit) * (1 ** self.n)) % self.n_square
                    )
            else:
                for digit in plaintext_as_digits_list:
                    r = PrimeDigit().generating_a_large_prime_modulo(self.n)
                    encrypt_text_as_digits_list.append(
                        ((self.g ** digit) * (r ** self.n)) % self.n_square
                    )
            return encrypt_text_as_digits_list
        else:
            return []


class PaillierPrivateKey(object):
    """Contains a private key and associated decryption method.

    Args:
        public's:
            lambdas (int): part of private key - see [1] \n
            mu (int): part of private key - see [1] \n
        private's:
            __public_key (object): object of class PaillierPublicKey
            __p (int): large prime
            __q (int): large prime

    Links:
        [1] - https://en.wikipedia.org/wiki/Paillier_cryptosystem#Key_generation
    """

    def __init__(self, public_key, p, q):
        self.__public_key: PaillierPublicKey = public_key
        self.__p = p
        self.__q = q
        self.lambdas = self.generation_lambdas(self.__p, self.__q)
        self.mu = self.generation_mu(self.__public_key.g, self.__public_key.n, self.lambdas)

    @staticmethod
    def generation_lambdas(p: int, q: int):
        """Function for generating lambdas as part of a private key.

        :param p: (int) large prime
        :param q: (int) large prime
        :return: (int) _lambdas as least common multiple of (p - 1) and (q - 1)
        """
        _lambdas = Euclid().least_common_multiple(p - 1, q - 1)
        return _lambdas

    @staticmethod
    def generation_mu(_g: int, _n: int, _lambdas: int):
        """Function for generating mu as part of a private key.

        :param _g: (int) part of public key
        :param _n: (int) part of public key
        :param _lambdas: (int) _lambdas as least common multiple of (p - 1) and (q - 1)
        :return: (int) mu as reverse digit modulo _n
        """
        result_l_func = l_func(pow(_g, _lambdas, _n ** 2), _n)
        reverse_digit = Euclid().reverse_digit(result_l_func, _n)
        mu = reverse_digit % _n
        return mu

    def show_private_key(self):
        """Private key display function

        :return: None
        """
        print(f"private_key: {self.lambdas}, {self.mu}")

    def decryption(self, encryption_digits_list: [int]):
        """Function for decrypting a list of encrypted numbers.

        :param encryption_digits_list: list [int] - list of encrypted numbers
        :return: list [int] - list of decrypted numbers
        """
        decrypt_text = []
        for encrypt_digit in encryption_digits_list:
            decrypt_digit = (
                    (l_func(
                        (encrypt_digit ** self.lambdas) % (self.__public_key.n ** 2),
                        self.__public_key.n
                    ) * self.mu) % self.__public_key.n
            )
            decrypt_text.append(decrypt_digit)

        return decrypt_text


class PaillierKeyPairGenerator(object):
    """Class includes function for generation public and private keys.

    """
    @staticmethod
    def paillier_key_pair_generation(bit_key_length: int = DEFAULT_BIT_KEY_LENGTH, return_pq: bool = False):
        """Function for generating public and private keys based on the bit length of the key.

        :param bit_key_length: (int) key length in bits (optional)
        :param return_pq: (bool) used for test
        :return: object's of classes PaillierPublicKey and PaillierPrivateKey
        """

        def p_q_generating(half_bit_key_length: int):
            """Helper function.
            Generates _p and _q as large primes by certain condition.

            :param half_bit_key_length: (int) half of key length in bits
            :return: (int) _p and (int) _q as large primes
            """

            _p = 3
            _q = 2

            while Euclid().greatest_common_divisor(_p * _q, (_p - 1) * (_q - 1)) != 1:
                _p = PrimeDigit().generation_a_large_prime(half_bit_key_length)
                _q = PrimeDigit().generation_a_large_prime(half_bit_key_length)
                while _q == _p:
                    _q = PrimeDigit().generation_a_large_prime(half_bit_key_length)

            return _p, _q

        p, q = p_q_generating(bit_key_length // 2)

        n = p * q

        public_key = PaillierPublicKey(n)
        private_key = PaillierPrivateKey(public_key, p, q)
        if return_pq:
            return public_key, private_key, p, q
        else:
            return public_key, private_key

    @staticmethod
    def paillier_key_pair_generation_from_pq(p: int, q: int):
        """Function for generating public and private keys based on numbers p and q.

        :param p: (int) large prime
        :param q: (int) large prime
        :return: object's of classes PaillierPublicKey and PaillierPrivateKey
        """
        n = p * q

        public_key = PaillierPublicKey(n)
        private_key = PaillierPrivateKey(public_key, p, q)

        return public_key, private_key


class Homomorphic(PaillierPublicKey):
    """

    """
    def __init__(self, n, g):
        super().__init__(n)
        self.g = g

    @staticmethod
    def comparison_of_text_lengths(
            first_encrypt_text_as_digits_list: [int],
            second_encrypt_text_as_digits_list: [int]
    ):
        """Helper function fo comparison of text lengths.

        :param first_encrypt_text_as_digits_list: (list [int])
        :param second_encrypt_text_as_digits_list: (list [int])
        :return: (bool) True if the text lengths are equivalent, False otherwise
        """
        length_first_enc_text = len(first_encrypt_text_as_digits_list)
        length_second_enc_text = len(second_encrypt_text_as_digits_list)
        if length_first_enc_text != length_second_enc_text:
            print("The texts are different in length.")
            if length_first_enc_text > length_second_enc_text:
                print("The first text is longer than the second. Add to the second text its values from the beginning "
                      "of the list, and then cut the second text to the length of the first in any way convenient for "
                      "you. ")
                print("Example (second_encrypt_text_as_digits_list * 2)[:len(first_encrypt_text_as_digits_list)]")
            else:
                print("The second text is longer than the first. Add to the first text its values from the beginning "
                      "of the list, and then cut the first text to the length of the second in any way convenient for "
                      "you. ")
                print("Example (first_encrypt_text_as_digits_list * 2)[:len(second_encrypt_text_as_digits_list)]")
            return False
        else:
            return True

    def addition_of_two_ciphertexts(
            self,
            first_encrypt_text_as_digits_list: [int],
            second_encrypt_text_as_digits_list: [int],
    ):
        """Function for adding two ciphertexts with one public key.

        :param first_encrypt_text_as_digits_list: (list [int])
        :param second_encrypt_text_as_digits_list: (list [int])
        :return: addition - (list [int] or list []) - the sum of two texts encrypted with one public key
                                                      or empty list if the text lengths are non-equivalent
        """

        addition = []

        if not self.comparison_of_text_lengths(first_encrypt_text_as_digits_list, second_encrypt_text_as_digits_list):
            print("An empty list will be returned to you.")
        else:
            for i in range(len(first_encrypt_text_as_digits_list)):
                addition.append(
                    (first_encrypt_text_as_digits_list[i] * second_encrypt_text_as_digits_list[i]) % self.n_square
                )
        return addition

    def addition_of_cipher_and_plaintext_via_g(
            self,
            first_encrypt_text_as_digits_list: [int],
            second_plaintext_as_digits_list: [int]
    ):
        """Function for adding encrypted and plaintext using one public key.

        :param first_encrypt_text_as_digits_list: (list [int])
        :param second_plaintext_as_digits_list: (list [int])
        :return: addition - (list [int] or list []) - the sum of two texts encrypted with one public key
                                                      or empty list if the text lengths are non-equivalent
        """

        addition = []

        if not self.comparison_of_text_lengths(first_encrypt_text_as_digits_list, second_plaintext_as_digits_list):
            print("An empty list will be returned to you.")
        else:
            second_encrypt_text_as_digits_list = self.encryption(second_plaintext_as_digits_list, True)
            for i in range(len(first_encrypt_text_as_digits_list)):
                addition.append(
                    (first_encrypt_text_as_digits_list[i] * second_encrypt_text_as_digits_list[i]) % self.n_square
                )
        return addition

    def __raising_an_encrypted_number_to_the_k_power(
            self,
            encrypted_number: int,
            k_power: int
    ):
        """Helper function for raising an encrypted number to the power k.

        :param encrypted_number: (int) - encrypted number
        :param k_power: (int) power
        :return: (int)
        """
        return pow(encrypted_number, k_power, self.n_square)

    def raising_of_ciphertext_to_the_power_of_plaintext(
            self,
            first_encrypt_text_as_digits_list: [int],
            second_plaintext_as_digits_list: [int]
    ):
        """Function of raising the ciphertext to the power of plaintext.

        :param first_encrypt_text_as_digits_list: (list [int])
        :param second_plaintext_as_digits_list: (list [int])
        :return: result_list - (list [int] or list []) - the result of raising the ciphertext to the power
                                                         of the plaintext with one public key
                                                         or empty list if the text lengths are non-equivalent
        """
        result_list = []

        if not self.comparison_of_text_lengths(first_encrypt_text_as_digits_list, second_plaintext_as_digits_list):
            print("An empty list will be returned to you.")
        else:
            for i in range(len(first_encrypt_text_as_digits_list)):
                result_list.append(
                    self.__raising_an_encrypted_number_to_the_k_power(
                        first_encrypt_text_as_digits_list[i], second_plaintext_as_digits_list[i]
                    )
                )
        return result_list

    def raising_the_ciphertext_to_the_k_power(
            self,
            encrypt_text_as_digits_list: [int],
            k_power: int
    ):
        """Function for raising the ciphertext to the power k.

        :param encrypt_text_as_digits_list: (list [int])
        :param k_power: (int) power
        :return: result_list - (list [int]) - the result of raising the ciphertext to the power k
        """

        result_list = []

        for i in range(len(encrypt_text_as_digits_list)):
            result_list.append(
                self.__raising_an_encrypted_number_to_the_k_power(
                    encrypt_text_as_digits_list[i], k_power
                )
            )
        return result_list
