from py_paillier.py_paillier import PaillierPublicKey, PaillierPrivateKey, Homomorphic
from py_paillier.py_paillier import PaillierKeyPairGenerator as pkpg
from unittest import main, TestCase


MIN_POWER = 1024
MAX_POWER = 1025


class PaillierKeyPairGenerationTest(TestCase):

    def test_default_create_key_pair(self):

        public_key, private_key = pkpg().paillier_key_pair_generation()

        self.assertTrue(hasattr(public_key, 'g'))
        self.assertTrue(hasattr(public_key, 'n'))

        self.assertTrue(hasattr(private_key, 'lambdas'))
        self.assertTrue(hasattr(private_key, 'mu'))

        self.assertTrue(str(public_key).startswith('<py_paillier.py_paillier.PaillierPublicKey '))
        self.assertTrue(str(private_key).startswith('<py_paillier.py_paillier.PaillierPrivateKey '))

    def test_create_key_pair(self):
        for power in range(MIN_POWER, MAX_POWER):

            public_key, private_key = pkpg().paillier_key_pair_generation(power)

            self.assertTrue(hasattr(public_key, 'g'))
            self.assertTrue(hasattr(public_key, 'n'))

            self.assertTrue(hasattr(private_key, 'lambdas'))
            self.assertTrue(hasattr(private_key, 'mu'))

            repr(public_key)
            repr(private_key)

    def test_public_key_constructor(self):
        for power in range(MIN_POWER, MAX_POWER):
            public_key, private_key = pkpg().paillier_key_pair_generation(power)
            public_key_from_static = PaillierPublicKey(public_key.n)

            self.assertEqual(public_key.n, public_key_from_static.n)

    def test_private_key_constructor(self):
        for power in range(MIN_POWER, MAX_POWER):
            public_key, private_key, p, q = pkpg().paillier_key_pair_generation(power, True)
            private_key_from_static = PaillierPrivateKey(public_key, p, q)

            self.assertEqual(private_key.lambdas, private_key_from_static.lambdas)
            self.assertEqual(private_key.mu, private_key_from_static.mu)


class PaillierEnDecryptionTest(TestCase):

    def test_on_modulo_n(self):
        public_key, private_key = pkpg().paillier_key_pair_generation()

        plaintext_1 = public_key.n - 1
        cipher_text_1 = public_key.encryption([plaintext_1])
        self.assertEqual([plaintext_1], private_key.decryption(cipher_text_1))

        plaintext_2 = public_key.n
        cipher_text_2 = public_key.encryption([plaintext_2])
        self.assertEqual([], private_key.decryption(cipher_text_2))

        plaintext_3 = public_key.n + 1
        cipher_text_3 = public_key.encryption([plaintext_3])
        self.assertEqual([], private_key.decryption(cipher_text_3))

    def test_en_decryption(self):
        public_key, private_key = pkpg().paillier_key_pair_generation(16)

        plaintext = [20094, 21774, 16518, 18209, 22329]
        encrypt_text = public_key.encryption(plaintext)
        decrypt_text = private_key.decryption(encrypt_text)

        self.assertEqual(plaintext, decrypt_text)

    def test_en_decryption_via_g(self):
        public_key, private_key = pkpg().paillier_key_pair_generation_from_pq(223, 211)

        plaintext = [19025, 32145, 17900, 29522, 30085]
        encrypt_text = public_key.encryption(plaintext, True)
        decrypt_text = private_key.decryption(encrypt_text)

        self.assertEqual(plaintext, decrypt_text)


class HomomorphicTest(TestCase):

    def test_default_constructor(self):
        public_key, private_key, p, q = pkpg().paillier_key_pair_generation(16, True)
        homomorphic = Homomorphic(public_key.n, public_key.g)
        # plaintext_1 = [20094, 25774, 16518, 18209, 22329]
        # plaintext_2 = [19025, 32145, 17900, 29522, 30085]

        self.assertEqual(public_key.n, homomorphic.n)
        self.assertEqual(public_key.n_square, homomorphic.n_square)

    def test_addition_of_two_ciphertexts(self):
        public_key, private_key = pkpg().paillier_key_pair_generation_from_pq(223, 211)
        homomorphic = Homomorphic(public_key.n, public_key.g)

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
        public_key, private_key = pkpg().paillier_key_pair_generation_from_pq(223, 211)
        homomorphic = Homomorphic(public_key.n, public_key.g)

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
        public_key, private_key = pkpg().paillier_key_pair_generation_from_pq(223, 211)
        homomorphic = Homomorphic(public_key.n, public_key.g)

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
        public_key, private_key = pkpg().paillier_key_pair_generation_from_pq(223, 211)
        homomorphic = Homomorphic(public_key.n, public_key.g)

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
        public_key, private_key = pkpg().paillier_key_pair_generation_from_pq(223, 211)
        homomorphic = Homomorphic(public_key.n, public_key.g)

        plaintext = [20094, 25774, 16518, 18209, 22329]
        cipher_text = public_key.encryption(plaintext)

        k_power = 18

        raising_of_ct_to_the_power_of_k = homomorphic.raising_the_ciphertext_to_the_k_power(cipher_text, k_power)

        default_multiple_plaintexts_to_k = [
            ((plaintext[i] * k_power) % public_key.n) for i in range(len(plaintext))
        ]

        decrypt_raising = private_key.decryption(raising_of_ct_to_the_power_of_k)

        self.assertEqual(default_multiple_plaintexts_to_k, decrypt_raising)


if __name__ == '__main__':
    main()
