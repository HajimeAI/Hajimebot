from solders.keypair import Keypair
from typing import List


def generate_keypair():
    return Keypair()

def restore_keypair(s: str):
    return Keypair.from_base58_string(s)
