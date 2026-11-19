import hashlib
import time
import json
import os

blockchain = []

# -------------------------
# SAVE BLOCKCHAIN TO FILE
# -------------------------
def save_blockchain():
    os.makedirs("server/logs", exist_ok=True)
    with open("server/logs/blockchain.json", "w") as f:
        json.dump(blockchain, f, indent=4)

# -------------------------
# ADD BLOCK
# -------------------------
def add_block(data):
    prev_hash = blockchain[-1]["hash"] if blockchain else "0"

    block_string = f"{prev_hash}{data}{time.time()}"
    block_hash = hashlib.sha256(block_string.encode()).hexdigest()

    block = {
        "data": data,
        "hash": block_hash
    }

    blockchain.append(block)

    print(f"[BLOCKCHAIN] Block added → {block_hash}")

    # 🔥 SAVE AFTER EVERY BLOCK
    save_blockchain()