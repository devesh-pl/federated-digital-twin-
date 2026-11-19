from cryptography.fernet import Fernet
import os
from dotenv import load_dotenv
import pickle

load_dotenv()

KEY = os.getenv("FERNET_KEY").encode()
fernet = Fernet(KEY)

def encrypt(data):
    serialized = pickle.dumps(data)
    return fernet.encrypt(serialized)

def decrypt(data):
    decrypted = fernet.decrypt(data)
    return pickle.loads(decrypted)