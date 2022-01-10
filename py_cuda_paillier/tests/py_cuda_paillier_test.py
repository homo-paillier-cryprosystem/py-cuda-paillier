from py_cuda_paillier.py_cuda_paillier import CudaConfig, PaillierPublicKey, PaillierPrivateKey, Homomorphic
from py_cuda_paillier.py_cuda_paillier import PaillierKeyPairGenerator as pkpg
import random
from unittest import main, TestCase


MIN_POWER = 10
MAX_POWER = 12


class PaillierKeyPairGenerationTest(TestCase):

    def test_default_create_key_pair(self):
        public_key: PaillierPublicKey
        private_key: PaillierPrivateKey
        public_key, private_key = pkpg().paillier_key_pair_generation()

        self.assertTrue(hasattr(public_key, 'g'))
        self.assertTrue(hasattr(public_key, 'n'))

        self.assertTrue(hasattr(private_key, 'lambdas'))
        self.assertTrue(hasattr(private_key, 'mu'))

        self.assertTrue(str(public_key).startswith('<py_cuda_paillier.py_cuda_paillier.PaillierPublicKey '))
        self.assertTrue(str(private_key).startswith('<py_cuda_paillier.py_cuda_paillier.PaillierPrivateKey '))

    def test_create_key_pair(self):
        for power in range(MIN_POWER, MAX_POWER):
            public_key: PaillierPublicKey
            private_key: PaillierPrivateKey
            public_key, private_key = pkpg().paillier_key_pair_generation(power)

            self.assertTrue(hasattr(public_key, 'g'))
            self.assertTrue(hasattr(public_key, 'n'))

            self.assertTrue(hasattr(private_key, 'lambdas'))
            self.assertTrue(hasattr(private_key, 'mu'))

            repr(public_key)
            repr(private_key)

    def test_public_key_constructor(self):
        for power in range(MIN_POWER, MAX_POWER):
            public_key: PaillierPublicKey
            private_key: PaillierPrivateKey
            public_key, private_key, p, q = pkpg().paillier_key_pair_generation(power, True)
            public_key_from_static = PaillierPublicKey(
                public_key.n, p, q
            )

            self.assertEqual(public_key.n, public_key_from_static.n)

    def test_private_key_constructor(self):
        for power in range(MIN_POWER, MAX_POWER):
            public_key: PaillierPublicKey
            private_key: PaillierPrivateKey
            public_key, private_key, p, q = pkpg().paillier_key_pair_generation(power, True)
            private_key_from_static = PaillierPrivateKey(public_key, p, q)

            self.assertEqual(private_key.lambdas, private_key_from_static.lambdas)
            self.assertEqual(private_key.mu, private_key_from_static.mu)


class PaillierEnDecryptionTest(TestCase):

    def test_on_modulo_n(self):
        public_key: PaillierPublicKey
        private_key: PaillierPrivateKey
        public_key, private_key = pkpg().paillier_key_pair_generation(MAX_POWER)

        plaintext_1 = [public_key.n - 1]
        cipher_text_1 = public_key.encryption(plaintext_1)
        self.assertEqual(plaintext_1, private_key.decryption(cipher_text_1))

        plaintext_2 = [public_key.n]
        cipher_text_2 = public_key.encryption(plaintext_2)
        self.assertEqual([], private_key.decryption(cipher_text_2))

        plaintext_3 = [public_key.n + 1]
        cipher_text_3 = public_key.encryption(plaintext_3)
        self.assertEqual([], private_key.decryption(cipher_text_3))

    def test_en_decryption(self):
        public_key: PaillierPublicKey
        private_key: PaillierPrivateKey
        public_key, private_key = pkpg().paillier_key_pair_generation(MAX_POWER)

        plaintext = [random.randint(1, public_key.n) for _ in range(5)]
        encrypt_text = public_key.encryption(plaintext)
        decrypt_text = private_key.decryption(encrypt_text)

        self.assertEqual(plaintext, decrypt_text)

    def test_en_decryption_via_g(self):
        public_key: PaillierPublicKey
        private_key: PaillierPrivateKey
        public_key, private_key = pkpg().paillier_key_pair_generation_from_pq(223, 211)

        plaintext = [19025, 32145, 17900, 29522, 30085]
        encrypt_text = public_key.encryption(plaintext, True)
        decrypt_text = private_key.decryption(encrypt_text)

        self.assertEqual(plaintext, decrypt_text)


class PaillierCudaEnDecryptionTest(TestCase):

    def test_cuda_en_decryption(self):
        public_key: PaillierPublicKey
        private_key: PaillierPrivateKey
        public_key, private_key, p, q = pkpg().paillier_key_pair_generation(MAX_POWER, True)

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
        self.assertEqual(plaintext, dec_text_from_gpu)

    def test_cuda_en_decryption_via_g(self):
        public_key: PaillierPublicKey
        private_key: PaillierPrivateKey
        public_key, private_key, p, q = pkpg().paillier_key_pair_generation(MAX_POWER, True)

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
            don_t_use_r=True
        )
        dec_text_from_gpu = private_key.cuda_decryption(gpu_enc_text, cuda_config)
        self.assertEqual(plaintext, dec_text_from_gpu)


class HomomorphicTest(TestCase):

    def test_default_constructor(self):
        for i in range(MIN_POWER, MAX_POWER + 1):
            public_key: PaillierPublicKey
            private_key: PaillierPrivateKey
            public_key, private_key, p, q = pkpg().paillier_key_pair_generation(i, True)
            homomorphic = Homomorphic(public_key.n, public_key.g, p, q)

            self.assertEqual(public_key.n, homomorphic.n)
            self.assertEqual(public_key.n_square, homomorphic.n_square)

    def test_addition_of_two_ciphertexts(self):
        p = 223
        q = 211
        public_key: PaillierPublicKey
        private_key: PaillierPrivateKey
        public_key, private_key = pkpg().paillier_key_pair_generation_from_pq(p, q)
        homomorphic = Homomorphic(public_key.n, public_key.g, p, q)

        plaintext_1 = [20094, 25774, 16518, 18209, 22329]
        cipher_text_1 = public_key.encryption(plaintext_1)

        plaintext_2 = [19025, 32145, 17900, 29522, 30085]
        cipher_text_2 = public_key.encryption(plaintext_2)

        addition_two_ciphertexts = homomorphic.addition_of_two_ciphertexts(
            cipher_text_1,
            cipher_text_2
        )

        default_addition_two_ciphertexts = [
            ((plaintext_1[i] + plaintext_2[i]) % public_key.n) for i in range(len(plaintext_1))
        ]

        decrypt_addition_of_two_cipher_text = private_key.decryption(addition_two_ciphertexts)

        self.assertEqual(default_addition_two_ciphertexts, decrypt_addition_of_two_cipher_text)

    def test_addition_of_ciphertext_and_ciphertext_via_g(self):
        public_key: PaillierPublicKey
        private_key: PaillierPrivateKey
        p = 223
        q = 211
        public_key, private_key = pkpg().paillier_key_pair_generation_from_pq(p, q)
        homomorphic = Homomorphic(public_key.n, public_key.g, p, q)

        plaintext_1 = [20094, 25774, 16518, 18209, 22329]
        cipher_text_1 = public_key.encryption(plaintext_1)

        plaintext_2 = [19025, 32145, 17900, 29522, 30085]
        cipher_text_2 = public_key.encryption(plaintext_2, True)

        addition_ciphertext_and_plaintext_via_g = homomorphic.addition_of_two_ciphertexts(
            cipher_text_1, cipher_text_2
        )

        default_addition_two_ciphertexts = [
            ((plaintext_1[i] + plaintext_2[i]) % public_key.n) for i in range(len(plaintext_1))
        ]

        decrypt_addition_of_two_cipher_text = private_key.decryption(addition_ciphertext_and_plaintext_via_g)

        self.assertEqual(default_addition_two_ciphertexts, decrypt_addition_of_two_cipher_text)

    def test_addition_of_cipher_and_plaintext_via_g(self):
        public_key: PaillierPublicKey
        private_key: PaillierPrivateKey
        p = 223
        q = 211
        public_key, private_key = pkpg().paillier_key_pair_generation_from_pq(p, q)
        homomorphic = Homomorphic(public_key.n, public_key.g, p, q)

        plaintext_1 = [20094, 25774, 16518, 18209, 22329]
        cipher_text_1 = public_key.encryption(plaintext_1)

        plaintext_2 = [19025, 32145, 17900, 29522, 30085]

        addition_ciphertext_and_plaintext_via_g = homomorphic.addition_of_cipher_and_plaintext_via_g(
            cipher_text_1, plaintext_2
        )

        default_addition_ciphertext_and_plaintext = [
            ((plaintext_1[i] + plaintext_2[i]) % public_key.n) for i in range(len(plaintext_1))
        ]

        decrypt_addition_ciphertext_and_plaintext_via_g = private_key.decryption(
            addition_ciphertext_and_plaintext_via_g
        )

        self.assertEqual(default_addition_ciphertext_and_plaintext, decrypt_addition_ciphertext_and_plaintext_via_g)

    def test_raising_of_ciphertext_to_the_power_of_plaintext(self):
        public_key: PaillierPublicKey
        private_key: PaillierPrivateKey
        p = 223
        q = 211
        public_key, private_key = pkpg().paillier_key_pair_generation_from_pq(p, q)
        homomorphic = Homomorphic(public_key.n, public_key.g, p, q)

        plaintext_1 = [20094, 25774, 16518, 18209, 22329]
        cipher_text_1 = public_key.encryption(plaintext_1)

        plaintext_2 = [19025, 32145, 17900, 29522, 30085]
        cipher_text_2 = public_key.encryption(plaintext_2)

        raising_of_ct_to_the_power_of_pt_1 = homomorphic.raising_of_ciphertext_to_the_power_of_plaintext(
            cipher_text_1, plaintext_2
        )

        raising_of_ct_to_the_power_of_pt_2 = homomorphic.raising_of_ciphertext_to_the_power_of_plaintext(
            cipher_text_2, plaintext_1
        )

        default_multiple_plaintexts = [
            ((plaintext_1[i] * plaintext_2[i]) % public_key.n) for i in range(len(plaintext_1))
        ]

        decrypt_raising_1 = private_key.decryption(raising_of_ct_to_the_power_of_pt_1)
        decrypt_raising_2 = private_key.decryption(raising_of_ct_to_the_power_of_pt_2)

        self.assertEqual(default_multiple_plaintexts, decrypt_raising_1)
        self.assertEqual(default_multiple_plaintexts, decrypt_raising_2)

    def test_raising_the_ciphertext_to_the_k_power(self):
        public_key: PaillierPublicKey
        private_key: PaillierPrivateKey
        p = 223
        q = 211
        public_key, private_key = pkpg().paillier_key_pair_generation_from_pq(p, q)
        homomorphic = Homomorphic(public_key.n, public_key.g, p, q)

        plaintext = [20094, 25774, 16518, 18209, 22329]
        cipher_text = public_key.encryption(plaintext)

        k_power = 18

        raising_of_ct_to_the_power_of_k = homomorphic.raising_the_ciphertext_to_the_k_power(cipher_text, k_power)

        default_multiple_plaintexts_to_k = [
            ((plaintext[i] * k_power) % public_key.n) for i in range(len(plaintext))
        ]

        decrypt_raising = private_key.decryption(raising_of_ct_to_the_power_of_k)

        self.assertEqual(default_multiple_plaintexts_to_k, decrypt_raising)


class HomomorphicCudaTest(TestCase):

    def test_cuda_addition_of_two_ciphertexts(self):
        public_key: PaillierPublicKey
        private_key: PaillierPrivateKey
        public_key, private_key, p, q = pkpg().paillier_key_pair_generation(return_pq=True)

        homomorphic = Homomorphic(public_key.n, public_key.g, p, q)

        cuda_config = CudaConfig(
            threads_per_block=100,
            blocks=200
        )

        plaintext_1 = [
            random.randrange(1, public_key.n) for _ in range(cuda_config.threads_per_block * cuda_config.blocks)
        ]

        gpu_enc_text_1 = public_key.cuda_encryption(
            plaintext_as_digits_list=plaintext_1,
            cuda_config=cuda_config,
            don_t_use_r=False
        )

        plaintext_2 = [
            random.randrange(1, public_key.n) for _ in range(cuda_config.threads_per_block * cuda_config.blocks)
        ]

        gpu_enc_text_2 = public_key.cuda_encryption(
            plaintext_as_digits_list=plaintext_2,
            cuda_config=cuda_config,
            don_t_use_r=False
        )

        addition_two_ciphertexts = homomorphic.cuda_addition_of_two_ciphertexts(
            first_encrypt_text_as_digits_list=gpu_enc_text_1,
            second_encrypt_text_as_digits_list=gpu_enc_text_2,
            cuda_config=cuda_config
        )

        default_addition_two_ciphertexts = [
            ((plaintext_1[i] + plaintext_2[i]) % public_key.n) for i in range(len(plaintext_1))
        ]

        decrypt_addition_of_two_cipher_text = private_key.cuda_decryption(
            encryption_digits_list=addition_two_ciphertexts,
            cuda_config=cuda_config
        )

        self.assertEqual(default_addition_two_ciphertexts, decrypt_addition_of_two_cipher_text)

    def test_cuda_addition_of_ciphertext_and_ciphertext_via_g(self):
        public_key: PaillierPublicKey
        private_key: PaillierPrivateKey
        public_key, private_key, p, q = pkpg().paillier_key_pair_generation(return_pq=True)

        homomorphic = Homomorphic(public_key.n, public_key.g, p, q)

        cuda_config = CudaConfig(
            threads_per_block=100,
            blocks=200
        )

        plaintext_1 = [
            random.randrange(1, public_key.n) for _ in range(cuda_config.threads_per_block * cuda_config.blocks)
        ]

        gpu_enc_text_1 = public_key.cuda_encryption(
            plaintext_as_digits_list=plaintext_1,
            cuda_config=cuda_config,
            don_t_use_r=False
        )

        plaintext_2 = [
            random.randrange(1, public_key.n) for _ in range(cuda_config.threads_per_block * cuda_config.blocks)
        ]

        addition_ciphertext_and_plaintext_via_g = homomorphic.cuda_addition_of_cipher_and_plaintext_via_g(
            first_encrypt_text_as_digits_list=gpu_enc_text_1,
            second_plaintext_as_digits_list=plaintext_2,
            cuda_config=cuda_config
        )

        default_addition_two_plaintexts = [
            ((plaintext_1[i] + plaintext_2[i]) % public_key.n) for i in range(len(plaintext_1))
        ]

        decrypt_addition_of_two_cipher_text = private_key.cuda_decryption(
            encryption_digits_list=addition_ciphertext_and_plaintext_via_g,
            cuda_config=cuda_config
        )

        self.assertEqual(default_addition_two_plaintexts, decrypt_addition_of_two_cipher_text)

    def test_cuda_raising_of_ciphertext_to_the_power_of_plaintext(self):
        public_key: PaillierPublicKey
        private_key: PaillierPrivateKey
        public_key, private_key, p, q = pkpg().paillier_key_pair_generation(return_pq=True)

        homomorphic = Homomorphic(public_key.n, public_key.g, p, q)

        cuda_config = CudaConfig(
            threads_per_block=100,
            blocks=200
        )

        plaintext_1 = [
            random.randrange(1, public_key.n) for _ in range(cuda_config.threads_per_block * cuda_config.blocks)
        ]

        gpu_enc_text_1 = public_key.cuda_encryption(
            plaintext_as_digits_list=plaintext_1,
            cuda_config=cuda_config,
            don_t_use_r=False
        )

        plaintext_2 = [
            random.randrange(1, public_key.n) for _ in range(cuda_config.threads_per_block * cuda_config.blocks)
        ]

        gpu_enc_text_2 = public_key.cuda_encryption(
            plaintext_as_digits_list=plaintext_2,
            cuda_config=cuda_config,
            don_t_use_r=False
        )

        raising_of_ct_to_the_power_of_pt_1 = homomorphic.cuda_raising_of_ciphertext_to_the_power_of_plaintext(
            gpu_enc_text_1, plaintext_2, cuda_config
        )

        raising_of_ct_to_the_power_of_pt_2 = homomorphic.cuda_raising_of_ciphertext_to_the_power_of_plaintext(
            gpu_enc_text_2, plaintext_1, cuda_config
        )

        default_multiple_plaintexts = [
            ((plaintext_1[i] * plaintext_2[i]) % public_key.n) for i in range(len(plaintext_1))
        ]

        decrypt_raising_1 = private_key.cuda_decryption(raising_of_ct_to_the_power_of_pt_1, cuda_config)
        decrypt_raising_2 = private_key.cuda_decryption(raising_of_ct_to_the_power_of_pt_2, cuda_config)

        self.assertEqual(default_multiple_plaintexts, decrypt_raising_1)
        self.assertEqual(default_multiple_plaintexts, decrypt_raising_2)

    def test_cuda_raising_the_ciphertext_to_the_k_power(self):
        public_key: PaillierPublicKey
        private_key: PaillierPrivateKey
        public_key, private_key, p, q = pkpg().paillier_key_pair_generation(return_pq=True)

        homomorphic = Homomorphic(public_key.n, public_key.g, p, q)

        cuda_config = CudaConfig(
            threads_per_block=100,
            blocks=200
        )

        plaintext = [
            random.randrange(1, public_key.n) for _ in range(cuda_config.threads_per_block * cuda_config.blocks)
        ]

        gpu_enc_text = public_key.cuda_encryption(
            plaintext_as_digits_list=plaintext,
            cuda_config=cuda_config,
            don_t_use_r=False
        )

        k_power = random.randint(1, public_key.n)

        raising_of_ct_to_the_power_of_k = homomorphic.cuda_raising_the_ciphertext_to_the_k_power(
            encrypt_text_as_digits_list=gpu_enc_text,
            k_power=k_power,
            cuda_config=cuda_config
        )

        default_multiple_plaintexts_to_k = [
            ((plaintext[i] * k_power) % public_key.n) for i in range(len(plaintext))
        ]

        decrypt_raising = private_key.cuda_decryption(
            encryption_digits_list=raising_of_ct_to_the_power_of_k,
            cuda_config=cuda_config
        )

        self.assertEqual(default_multiple_plaintexts_to_k, decrypt_raising)


class InputTypesTest(TestCase):

    def test_cuda_config(self):
        threads_types = [1, 1.0, '1', [1]]
        blocks_types = [2, 2.0, '2', [2]]

        for threads_type in threads_types:
            for blocks_type in blocks_types:
                cuda_config = CudaConfig(
                    threads_per_block=threads_type,
                    blocks=blocks_type
                )

                self.assertEqual(type(cuda_config.threads_per_block), int)
                self.assertEqual(type(cuda_config.blocks), int)

    def test_public_key(self):
        n_types = [2, 2.0, '2', [2]]
        p_types = [3, 3.0, '3', [3]]
        q_types = [4, 4.0, '4', [4]]

        for n_type in n_types:
            for p_type in p_types:
                for q_type in q_types:
                    public_key = PaillierPublicKey(
                        n=n_type, p=p_type, q=q_type
                    )

                    self.assertEqual(type(public_key.n), int)
                    self.assertEqual(type(public_key.g), int)

    def test_encryption(self):
        plaintext = [
            1, 2.0, '3', False
        ]

        public_key, private_key = pkpg.paillier_key_pair_generation()

        enc_text = public_key.encryption(plaintext)

        self.assertEqual([], enc_text)

    def test_cuda_encryption(self):
        plaintext = [
            1, 2.0, '3', False, True
        ]

        public_key, private_key = pkpg.paillier_key_pair_generation()

        cuda_config = CudaConfig(1, 5)

        enc_text = public_key.cuda_encryption(plaintext, cuda_config)

        self.assertEqual([], enc_text)

    def test_private_key(self):
        n_types = [2.0, '2', [2]]
        p_types = [3.0, '3', [3]]
        q_types = [4.0, '4', [4]]

        for n_type in n_types:
            for p_type in p_types:
                for q_type in q_types:
                    print(type(n_type), type(p_type), type(q_type))
                    public_key = PaillierPublicKey(
                        n=n_type, p=p_type, q=q_type
                    )

                    private_key = PaillierPrivateKey(
                        public_key=public_key, p=p_type, q=q_type
                    )

                    self.assertEqual(type(private_key.lambdas), int)
                    self.assertEqual(type(private_key.mu), int)

    def test_key_pair_generations(self):
        bit_key_length_types = [10, 10.0, '10']
        return_pq_types = [10, 10.0, '10', False]

        for bit_key_length_type in bit_key_length_types:
            for return_pq_type in return_pq_types:
                public_key, private_key = pkpg.paillier_key_pair_generation(
                    bit_key_length=bit_key_length_type,
                    return_pq=return_pq_type
                )

                self.assertEqual(type(public_key.n), int)
                self.assertEqual(type(public_key.g), int)
                self.assertEqual(type(private_key.lambdas), int)
                self.assertEqual(type(private_key.mu), int)

        p_types = [3.0, '3', [3]]
        q_types = [4.0, '4', [4]]
        for p_type in p_types:
            for q_type in q_types:
                public_key, private_key = pkpg.paillier_key_pair_generation_from_pq(
                    p=p_type, q=q_type
                )

                self.assertEqual(type(public_key.n), int)
                self.assertEqual(type(public_key.g), int)
                self.assertEqual(type(private_key.lambdas), int)
                self.assertEqual(type(private_key.mu), int)


if __name__ == '__main__':
    main()
