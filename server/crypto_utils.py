from cryptography.fernet import Fernet
import os
from dotenv import load_dotenv
import pickle

load_dotenv()

KEY = os.getenv("FERNET_KEY").encode()
fernet = Fernet(KEY)

def decrypt(data):
    return pickle.loads(fernet.decrypt(data))