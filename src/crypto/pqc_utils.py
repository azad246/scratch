import os
import pickle
import numpy as np

import ctypes.util

# oqs calls sys.exit() at import time if the native liboqs DLL is missing.
# Avoid that crash by checking for the shared library BEFORE importing.
_liboqs_found = ctypes.util.find_library("oqs") is not None
if _liboqs_found:
    try:
        import oqs
    except Exception:
        oqs = None
else:
    oqs = None


try:
    from dilithium_py.dilithium import Dilithium2
except ImportError:
    Dilithium2 = None


class PQCManager:
    def __init__(self, kem_alg="Kyber512"):
        self.kem_alg = kem_alg
        self.kem = oqs.KeyEncapsulation(kem_alg) if oqs else None

    def generate_kem_keypair(self):
        if self.kem is None:
            raise ImportError("liboqs-python is not installed.")
        public_key = self.kem.generate_keypair()
        secret_key = self.kem.export_secret_key()
        return public_key, secret_key

    def encapsulate(self, public_key):
        if self.kem is None:
            raise ImportError("liboqs-python is not installed.")
        ciphertext, shared_secret = self.kem.encap_secret(public_key)
        return ciphertext, shared_secret

    def decapsulate(self, ciphertext, secret_key=None):
        if self.kem is None:
            raise ImportError("liboqs-python is not installed.")
        shared_secret = self.kem.decap_secret(ciphertext)
        return shared_secret

    @staticmethod
    def serialize_weights(weights):
        return pickle.dumps(weights)

    @staticmethod
    def deserialize_weights(blob):
        return pickle.loads(blob)

    @staticmethod
    def xor_encrypt(data_bytes, key_bytes):
        key = key_bytes * (len(data_bytes) // len(key_bytes) + 1)
        return bytes([b ^ k for b, k in zip(data_bytes, key)])

    @staticmethod
    def xor_decrypt(data_bytes, key_bytes):
        return PQCManager.xor_encrypt(data_bytes, key_bytes)


class SignatureManager:
    def __init__(self):
        if Dilithium2 is None:
            raise ImportError("dilithium-py is not installed.")

    def generate_keypair(self):
        pk, sk = Dilithium2.keygen()
        return pk, sk

    def sign(self, sk, msg: bytes):
        return Dilithium2.sign(sk, msg)

    def verify(self, pk, msg: bytes, sig: bytes):
        return Dilithium2.verify(pk, msg, sig)
