"""Help functions for py_paillier"""

import random


class Euclid(object):
    """Euclid's algorithms"""

    @staticmethod
    def greatest_common_divisor(a: int, b: int):
        """Function wrapper for function gcd.
        :param a: (int)
        :param b: (int)
        :return: (int) greatest common divisor
        """
        def gcd(_a: int, _b: int):
            """The function of calculating the greatest common divisor.

            :param _a: (int)
            :param _b: (int)
            :return: (int) greatest common divisor
            """
            q = _a // _b
            r = _a - _b * q
            if r == 0:
                return _b
            else:
                return gcd(_b, r)

        if a < b:
            a, b = b, a

        return gcd(a, b)

    @staticmethod
    def least_common_multiple(a: int, b: int):
        """The function of calculating the least common multiple.

        :param a: (int)
        :param b: (int)
        :return: (int) least common multiple
        """
        gsd_result = Euclid().greatest_common_divisor(a, b)
        return int((a * b) / gsd_result)

    @staticmethod
    def reverse_digit(a: int, n: int):
        """The function of calculating the reverse digit (Unit [see [1]]) modulo n.

        :param a: (int)
        :param n: (int) as modulo
        :return: (int) reverse digit (Unit)

        Links:
            [1] - https://en.wikipedia.org/wiki/Unit_(ring_theory)
        """

        x, xx, y, yy = 1, 0, 0, 1
        _max = max(a, n)
        while n:
            q = a // n
            r = a % n
            a, n = n, r
            if r == 0:
                return xx
            else:
                x, xx = xx % _max, (x + xx * (_max - q)) % _max
                y, yy = yy % _max, (y + yy * (_max - q)) % _max


class PrimeDigit(object):

    @staticmethod
    def fermat_s_little_theorem(n: int):
        """Function for checking simplicity by Fermat's little theorem - see [1].

        :param n: (int) digit for test
        :return: (bool) prime number (True) or not a prime number (False)

        Links:
            [1] - https://en.wikipedia.org/wiki/Fermat%27s_little_theorem
        """
        if pow(2, n - 1, n) == 1:
            return True
        else:
            return False

    @staticmethod
    def generation_a_large_prime_by_search(n: int):
        """The function of generating a large simply number by searching for the next from random.

        :param n: (int) key length in bits
        :return: (int) large prime
        """

        digit = random.SystemRandom().randrange(
            2 ** (n - 1),
            2 ** n
        )
        if digit % 2 == 0:
            digit += 1
        while not PrimeDigit().fermat_s_little_theorem(digit):
            digit += 2
        return digit

    @staticmethod
    def generation_a_large_prime(n: int):
        """Large prime generation function for p and q.

        :param n: (int) key length in bits
        :return: (int) large prime
        """
        x = 4
        while not PrimeDigit().fermat_s_little_theorem(x):
            x = random.SystemRandom().randrange(
                2 ** (n - 1),
                2 ** n
            )
        return x

    @staticmethod
    def generating_a_large_prime_modulo(n: int):
        """Large prime generation function for g in modulo n.

        :param n: (int) as modulo
        :return: (int) large prime
        """
        x = 4
        while not PrimeDigit().fermat_s_little_theorem(x):
            x = random.SystemRandom().randrange(n // 2, n)
        return x

    @staticmethod
    def sieve_of_eratosthenes(n: int):
        """Function of finding all primes up to some integer n.

        :param n: (int) as limit
        :return: (list[int]) list of primes up to n

        Links:
            [1] - https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
        """

        list_all_numbers = list(range(n + 1))
        list_all_numbers[1] = 0
        primes_list = []
        i = 2
        while i <= n:
            if list_all_numbers[i] != 0:
                primes_list.append(list_all_numbers[i])
                for j in range(i, n + 1, i):
                    list_all_numbers[j] = 0
            i += 1
        return primes_list

    @staticmethod
    def segment_sieve_of_eratosthenes(n: int):
        """Function of finding all prime numbers up to some integer n by segments.

        :param n: (int) as limit
        :return: (list[int]) list of primes up to n

        Links:
            [1] - https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
        """

        def create_next_segment_as_bool_list(_n: int, _down_limit: int, _delta: int):
            """

            :param _n: (int) as limit
            :param _down_limit: (int)
            :param _delta: the difference between the down_limit and up_limit
            :return: (list[bool]) - boolean list for less memory
            """
            if (_down_limit + _delta) > _n:
                up_limit = _n
            else:
                up_limit = _down_limit + _delta
            return list(True for _ in range(_down_limit, up_limit))

        delta = 100000
        down_limit = delta

        list_all_numbers = list(range(delta))
        list_all_numbers[1] = 1
        primes_list = []
        i = 2
        while i < delta:
            if list_all_numbers[i] != 0:
                primes_list.append(list_all_numbers[i])
                for j in range(i, delta, i):
                    list_all_numbers[j] = 0
            i += 1

        while down_limit < n:
            segment = create_next_segment_as_bool_list(n, down_limit, delta)

            if (down_limit + delta) >= n:
                last_segment_element = n - 1
            else:
                last_segment_element = down_limit + delta - 1

            for prime in primes_list:
                if pow(prime, 2) < last_segment_element:
                    for index_element in range(len(segment) - 1):
                        true_index = down_limit + index_element
                        if true_index % prime == 0:
                            for to_false_index in range(true_index, last_segment_element, true_index):
                                segment[true_index - down_limit] = False

            for index_element in range(len(segment) - 1):
                if segment[index_element]:
                    true_index = down_limit + index_element
                    primes_list.append(true_index)

            down_limit += delta

        return primes_list


def calc_reduced_system_deductions(n: int):
    """Function for calculating the reduced system of residues modulo n.

    :param n: (int) as modulo
    :return: (list[int]) list of numbers of the reduced system of residues modulo n
    """

    multiplicative_group = PrimeDigit().sieve_of_eratosthenes(n)

    return multiplicative_group


def check_plaintext(plaintext_as_digits_list: [int], n: int):
    """Function of checking the text for the possibility of encryption.

    :param plaintext_as_digits_list: (list[int]) - list of unencrypted numbers
    :param n: (int) as limit
    :return: (bool) suitable text (True) or not (False)
    """
    currently = True

    unique_digits = list(set(plaintext_as_digits_list))
    digits_not_in_n = []

    for digit in unique_digits:
        if digit not in range(n):
            digits_not_in_n.append(digit)
            currently = False

    if not currently:
        print(f"\nThe given numbers {digits_not_in_n} do not belong to the set  Z_{n}")
        print("Re-generate keys for more bits.")
        return False
    else:
        print("\nThe entered text has been successfully verified.")
        return True
