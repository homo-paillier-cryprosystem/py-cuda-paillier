"""Paillier encryption library for partially homomorphic encryption."""

from py_cuda_paillier.util import Euclid, PrimeDigit, check_plaintext
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
import numpy as np

DEFAULT_BIT_KEY_LENGTH = 12


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


class CudaConfig(object):
    """Class for setting parameters of the GPU.

    Args:
        :arg threads_per_block (int): number of threads in a block
        :arg blocks (int): number of blocks in the grid
    """
    def __init__(self, threads_per_block: int, blocks: int):
        if isinstance(threads_per_block, int) and isinstance(blocks, int):
            self.threads_per_block: int = threads_per_block
            self.blocks: int = blocks
        else:
            print("The input data (threads_per_block or blocks) is of an invalid type.")
            print("Class parameters value threads_per_block and blocks were set to 1.")
            self.threads_per_block = 1
            self.blocks = 1


class PaillierPublicKey(object):
    """Contains a public key and associated encryption method.

    Args:
        :arg n (int): part of public key - see [1] \n
        :arg g (int): part of public key - see [1] \n
        :arg n_square (int): (n ** 2), stored for calculations \n

    Links:
        [1] - https://en.wikipedia.org/wiki/Paillier_cryptosystem#Key_generation
    """

    def __init__(self, n: int, p: int, q: int):
        if isinstance(n, int) and isinstance(p, int) and isinstance(q, int):
            # public key
            self.n = n
            self.g = self.generation_g(self.n)
            # parameters for calculations
            self.n_square = self.n ** 2
            self.__p = p
            self.__q = q
        else:
            print("The input data (n, p or q) is of an invalid type.")
            # public key
            self.n = 1517
            self.g = self.generation_g(self.n)
            # parameters for calculations
            self.n_square = self.n ** 2
            self.__p = 41
            self.__q = 37
            print("Class parameters value were set to:")
            print(f"\tn = {self.n}")
            print(f"\tg = {self.g}")

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

        :param plaintext_as_digits_list: (list[int]) - list of unencrypted numbers
        :param don_t_use_r: (bool) optional, used for homomorphic encryption function
        :return: empty list [] or non-empty list including encrypted digits
        """
        if not isinstance(don_t_use_r, bool):
            print("The parameter 'don_t_use_r' did not match a boolean type, so its value was set to False.")
            don_t_use_r = False
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
                    r = PrimeDigit().gen_mutually(self.n)
                    encrypt_text_as_digits_list.append(
                        ((self.g ** digit) * (r ** self.n)) % self.n_square
                    )
            return encrypt_text_as_digits_list
        else:
            return []

    def cuda_encryption(
            self, plaintext_as_digits_list: [int],
            cuda_config: CudaConfig, don_t_use_r: bool = False
    ):
        """Encryption function of plaintext represented as a list of numbers using the GPU.

        :param plaintext_as_digits_list: (list[int]) - list of unencrypted numbers
        :param cuda_config: (list[int, int]) - parameters for GPU
        :param don_t_use_r: (bool) optional, used for homomorphic encryption function
        :return: empty list [] or non-empty list including encrypted digits
        """
        @cuda.jit
        def rand_array(p: int, q: int, _rng_states, out: np.ndarray):
            thread_id = cuda.grid(1)
            x = xoroshiro128p_uniform_float64(_rng_states, thread_id) * 100000
            while int(x) % p == 0 or int(x) % q == 0 or int(x) > (p * q):
                x = xoroshiro128p_uniform_float64(_rng_states, thread_id) * 100000
            out[thread_id] = x

        @cuda.jit
        def calculation(
                n: int, g: int, n_square: int, _np_randoms: np.ndarray,
                np_plaintext: np.ndarray, _np_enc_text: np.ndarray
        ):
            thread_id = cuda.grid(1)

            g_pow = g
            mod = n_square
            i = 1
            while i != np_plaintext[thread_id]:
                g_pow = g_pow * g
                if g_pow > mod:
                    g_pow = g_pow % mod
                i += 1

            r_pow = _np_randoms[thread_id]
            mod = n_square
            i = 1
            while i != n:
                r_pow = r_pow * _np_randoms[thread_id]
                if r_pow > mod:
                    r_pow = r_pow % mod
                i += 1

            _np_enc_text[thread_id] = (g_pow * r_pow) % n_square

        if not isinstance(don_t_use_r, bool):
            print("The parameter 'don_t_use_r' did not match a boolean type, so its value was set to False.")
            don_t_use_r = False
        if not isinstance(cuda_config, CudaConfig):
            print("The parameter 'cuda_config' did not match a boolean type, so its value was set to [1, 1].")
            cuda_config = CudaConfig(1, 1)
        plaintext_is_current: bool = check_plaintext(
            plaintext_as_digits_list, self.n
        )
        if plaintext_is_current:
            randoms = np.ones(len(plaintext_as_digits_list), dtype=np.uint64)
            # generate randoms digits with cuda
            if not don_t_use_r:
                # generate states
                rng_states = create_xoroshiro128p_states(len(plaintext_as_digits_list), seed=1)
                # create numpy array for random
                np_randoms = np.ones(len(plaintext_as_digits_list), dtype=np.float64)
                # copy array to GPU
                device_np_randoms = cuda.to_device(np_randoms)
                # generate randoms
                rand_array[cuda_config.blocks, cuda_config.threads_per_block](
                    self.__p, self.__q,
                    rng_states,
                    device_np_randoms
                )
                # copy array to CPU
                host_np_randoms = device_np_randoms.copy_to_host()
                cuda.close()
                # transfer from numpy
                randoms = host_np_randoms.astype(int).tolist()
            # encryption part
            # transfer plaintext to numpy array
            np_plaintext_as_dl = np.array(plaintext_as_digits_list, dtype=np.uint64)
            np_randoms = np.array(randoms, dtype=np.uint64)

            # create numpy array for encrypt text
            np_enc_text = np.ones_like(np_plaintext_as_dl, dtype=np.uint64)

            # copy numpy arrays to GPU
            device_nnp_randoms = cuda.to_device(np_randoms)
            device_np_plaintext_as_dl = cuda.to_device(np_plaintext_as_dl)
            device_np_enc_text = cuda.to_device(np_enc_text)

            # encrypt
            calculation[cuda_config.blocks, cuda_config.threads_per_block](
                self.n, self.g, self.n_square,
                device_nnp_randoms,
                device_np_plaintext_as_dl,
                device_np_enc_text
            )

            # copy array to host
            host_enc_text = device_np_enc_text.copy_to_host()
            cuda.close()

            # change types
            encrypt_text_as_digits_list = host_enc_text.astype(int).tolist()
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

    def __init__(self, public_key: PaillierPublicKey, p: int, q: int):
        if isinstance(public_key, PaillierPublicKey) and isinstance(p, int) and isinstance(q, int):
            self.__public_key: PaillierPublicKey = public_key
            self.__p = p
            self.__q = q
        else:
            print("The input data (public_key, p or q) is of an invalid type.")
            self.__public_key: PaillierPublicKey = PaillierPublicKey(1517, 41, 38)
            self.__p = 41
            self.__q = 38

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
                        pow(encrypt_digit, self.lambdas, self.__public_key.n_square),
                        self.__public_key.n
                    ) * self.mu) % self.__public_key.n
            )
            decrypt_text.append(decrypt_digit)

        return decrypt_text

    def cuda_decryption(self, encryption_digits_list: [int], cuda_config: CudaConfig):
        """Function for decrypting a list of encrypted numbers with GPU.

        :param encryption_digits_list: (list [int]) - list of encrypted numbers
        :param cuda_config: (list[int, int]) - parameters for GPU
        :return: list [int] - list of decrypted numbers
        """
        @cuda.jit
        def calculation(
                lambdas, mu, n, n_square,
                enc_text, dec_text
        ):
            thread_id = cuda.grid(1)

            pow_enc_text = enc_text[thread_id]
            mod = n_square
            i = 1
            while i != lambdas:
                pow_enc_text = pow_enc_text * enc_text[thread_id]
                if pow_enc_text > mod:
                    pow_enc_text = pow_enc_text % mod
                i += 1

            res_l_func = (pow_enc_text - 1) // n

            dec_text[thread_id] = (res_l_func * mu) % n

        if not isinstance(cuda_config, CudaConfig):
            print("The parameter 'cuda_config' did not match a boolean type, so its value was set to [1, 1].")
            cuda_config = CudaConfig(1, 1)

        np_encryption_digits_list = np.array(encryption_digits_list, dtype=np.uint64)
        device_np_enc_dl = cuda.to_device(np_encryption_digits_list)

        np_decrypt_text = np.ones_like(np_encryption_digits_list)
        device_decrypt_text = cuda.to_device(np_decrypt_text)

        calculation[cuda_config.blocks, cuda_config.threads_per_block](
            self.lambdas, self.mu, self.__public_key.n, self.__public_key.n_square,
            device_np_enc_dl, device_decrypt_text
        )

        np_decrypt_text = device_decrypt_text.copy_to_host()
        decrypt_text = np_decrypt_text.astype(int).tolist()

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

            while Euclid().greatest_common_divisor(_p * _q, (_p - 1) * (_q - 1)) != 1 or (_p * _q) > 9798:
                _p = PrimeDigit().generation_a_large_prime(half_bit_key_length)
                _q = PrimeDigit().generation_a_large_prime(half_bit_key_length)
                while _q == _p:
                    _q = PrimeDigit().generation_a_large_prime(half_bit_key_length)

            return _p, _q

        if not isinstance(bit_key_length, int):
            print("Parameter 'bit_key_length' must be type 'int', set value 12.")
            bit_key_length = DEFAULT_BIT_KEY_LENGTH
        if not isinstance(return_pq, bool):
            print("Parameter 'return_pq' must be type 'bool', set value False.")
            return_pq = False

        p, q = p_q_generating(bit_key_length // 2)

        n = p * q

        public_key = PaillierPublicKey(n, p, q)
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
        if not isinstance(p, int):
            print("Parameter 'p' must be type 'int', set value 41.")
            p = 41
        if not isinstance(q, int):
            print("Parameter 'q' must be type 'int', set value 38.")
            q = 38

        n = p * q

        public_key = PaillierPublicKey(n, p, q)
        private_key = PaillierPrivateKey(public_key, p, q)

        return public_key, private_key


class Homomorphic(PaillierPublicKey):
    """Class of Homomorphic Properties.

    """
    def __init__(self, n, g, p, q):
        super().__init__(n, p, q)
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
            second_encrypt_text_as_digits_list: [int]
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

    def cuda_addition_of_two_ciphertexts(
            self,
            first_encrypt_text_as_digits_list: [int],
            second_encrypt_text_as_digits_list: [int],
            cuda_config: CudaConfig
    ):
        """Function for adding two ciphertexts with one public key using GPU.

        :param first_encrypt_text_as_digits_list: (list [int])
        :param second_encrypt_text_as_digits_list: (list [int])
        :param cuda_config: (list [int, int]) object of CudaConfig
        :return: addition - (list [int] or list []) - the sum of two texts encrypted with one public key
                                                      or empty list if the text lengths are non-equivalent
        """

        @cuda.jit
        def calculation(
                n_square, np_f_enc, np_s_enc, np_adds
        ):
            thread_id = cuda.grid(1)
            np_adds[thread_id] = (np_f_enc[thread_id] * np_s_enc[thread_id]) % n_square

        addition = []

        if not self.comparison_of_text_lengths(first_encrypt_text_as_digits_list, second_encrypt_text_as_digits_list):
            print("An empty list will be returned to you.")
        else:

            np_addition = np.ones(len(first_encrypt_text_as_digits_list), dtype=np.uint64)
            np_first_enc = np.array(first_encrypt_text_as_digits_list, dtype=np.uint64)
            np_second_enc = np.array(second_encrypt_text_as_digits_list, dtype=np.uint64)

            # copy arrays to GPU
            device_np_addition = cuda.to_device(np_addition)
            device_np_first_enc = cuda.to_device(np_first_enc)
            device_np_second_enc = cuda.to_device(np_second_enc)

            calculation[cuda_config.blocks, cuda_config.threads_per_block](
                self.n_square, device_np_first_enc,
                device_np_second_enc, device_np_addition
            )

            host_np_addition = device_np_addition.copy_to_host()
            addition = host_np_addition.astype(int).tolist()

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

    def cuda_addition_of_cipher_and_plaintext_via_g(
            self,
            first_encrypt_text_as_digits_list: [int],
            second_plaintext_as_digits_list: [int],
            cuda_config: CudaConfig
    ):
        """Function for adding encrypted and plaintext using one public key using GPU.

        :param first_encrypt_text_as_digits_list: (list [int])
        :param second_plaintext_as_digits_list: (list [int])
        :param cuda_config: (list [int, int])
        :return: addition - (list [int] or list []) - the sum of two texts encrypted with one public key
                                                      or empty list if the text lengths are non-equivalent
        """

        addition = []

        if not self.comparison_of_text_lengths(first_encrypt_text_as_digits_list, second_plaintext_as_digits_list):
            print("An empty list will be returned to you.")
        else:
            second_encrypt_text_as_digits_list = self.cuda_encryption(
                second_plaintext_as_digits_list,
                cuda_config,
                don_t_use_r=True
            )
            addition = self.cuda_addition_of_two_ciphertexts(
                first_encrypt_text_as_digits_list,
                second_encrypt_text_as_digits_list,
                cuda_config
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

    def cuda_raising_of_ciphertext_to_the_power_of_plaintext(
            self,
            first_encrypt_text_as_digits_list: [int],
            second_plaintext_as_digits_list: [int],
            cuda_config: CudaConfig
    ):
        """Function of raising the ciphertext to the power of plaintext using GPU.

        :param first_encrypt_text_as_digits_list: (list [int])
        :param second_plaintext_as_digits_list: (list [int])
        :param cuda_config: (list [int, int])
        :return: result_list - (list [int] or list []) - the result of raising the ciphertext to the power
                                                         of the plaintext with one public key
                                                         or empty list if the text lengths are non-equivalent
        """

        @cuda.jit
        def calculation(np_f_enc, np_s_plain, n_square, np_adds):
            thread_id = cuda.grid(1)
            enc_pow = np_f_enc[thread_id]
            mod = n_square
            i = 1
            while i != np_s_plain[thread_id]:
                enc_pow = enc_pow * np_f_enc[thread_id]
                if enc_pow > mod:
                    enc_pow = enc_pow % mod
                i += 1

            np_adds[thread_id] = enc_pow

        result_list = []

        if not self.comparison_of_text_lengths(first_encrypt_text_as_digits_list, second_plaintext_as_digits_list):
            print("An empty list will be returned to you.")
        else:
            np_result = np.ones(len(first_encrypt_text_as_digits_list), dtype=np.uint64)
            np_first_enc = np.array(first_encrypt_text_as_digits_list, dtype=np.uint64)
            np_second_plain = np.array(second_plaintext_as_digits_list, dtype=np.uint64)

            # copy arrays to GPU
            device_np_result = cuda.to_device(np_result)
            device_np_first_enc = cuda.to_device(np_first_enc)
            device_np_second_plain = cuda.to_device(np_second_plain)

            calculation[cuda_config.blocks, cuda_config.threads_per_block](
                device_np_first_enc, device_np_second_plain,
                self.n_square, device_np_result
            )

            host_np_result = device_np_result.copy_to_host()
            result_list = host_np_result.astype(int).tolist()

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

    def cuda_raising_the_ciphertext_to_the_k_power(
            self,
            encrypt_text_as_digits_list: [int],
            k_power: int,
            cuda_config: CudaConfig
    ):
        """Function for raising the ciphertext to the power k using GPU.

        :param encrypt_text_as_digits_list: (list [int])
        :param k_power: (int) power
        :param cuda_config: (list [int, int])
        :return: result_list - (list [int]) - the result of raising the ciphertext to the power k
        """

        @cuda.jit
        def calculation(np_f_enc, _k_power, n_square, np_adds):
            thread_id = cuda.grid(1)
            enc_pow = np_f_enc[thread_id]
            mod = n_square
            i = 1
            while i != _k_power:
                enc_pow = enc_pow * np_f_enc[thread_id]
                if enc_pow > mod:
                    enc_pow = enc_pow % mod
                i += 1

            np_adds[thread_id] = enc_pow

        np_result = np.ones(len(encrypt_text_as_digits_list), dtype=np.uint64)
        np_first_enc = np.array(encrypt_text_as_digits_list, dtype=np.uint64)

        # copy arrays to GPU
        device_np_result = cuda.to_device(np_result)
        device_np_first_enc = cuda.to_device(np_first_enc)

        calculation[cuda_config.blocks, cuda_config.threads_per_block](
            device_np_first_enc, k_power,
            self.n_square, device_np_result
        )

        host_np_result = device_np_result.copy_to_host()
        result_list = host_np_result.astype(int).tolist()

        return result_list
