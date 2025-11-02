# des_module.py
import base64
import hashlib
from Crypto.Cipher import DES  # type: ignore
from Crypto.Util.Padding import pad, unpad  # type: ignore

def derive_des_key(user_input: str) -> bytes:
    md5_hash = hashlib.md5(user_input.encode()).digest()
    return md5_hash[:8]  # DES key must be exactly 8 bytes

def encrypt_des(data: str, key: bytes) -> str:
    cipher = DES.new(key, DES.MODE_ECB)
    ciphertext = cipher.encrypt(pad(data.encode(), DES.block_size))
    ciphertext_b64 = base64.b64encode(ciphertext).decode()
    return ciphertext_b64

def decrypt_des(ciphertext_b64: str, key: bytes) -> str:
    ciphertext = base64.b64decode(ciphertext_b64)
    cipher = DES.new(key, DES.MODE_ECB)
    plaintext = unpad(cipher.decrypt(ciphertext), DES.block_size)
    return plaintext.decode()
