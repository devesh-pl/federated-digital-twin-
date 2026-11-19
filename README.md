
federated-digital-twin


The project is a federated time-series classification demo. Its code trains a shared model without sending each hospital's training rows as part
of the normal client upload.

Each hospital runs a client process. The client reads that hospital's data, trains a local model, and sends model parameters
to a central server. The server combines the parameters, evaluates the combined model, saves artifacts, and makes the
model available to the clients. Each hospital then checks whether a personalized version is acceptable for its own data.

1 LOCAL DATA
Each hospital keeps
its own CSV
2 CLIENT TRAINING
Train and score a
local LSTM
3 SERVER
Decrypt, average,
evaluate and save
4 PERSONALIZE
Receive global model,
select and repeat

The model and how it learns:
This section translates the main machine-learning terms into the project's implementation.
The shared model is an LSTM, a neural network design that can read ordered sequences. Each time step has six values.
The model reads the sequence, takes the output from the last time step, and produces four raw scores, one for each
severity class. Softmax converts those scores into probabilities when the code evaluates or predicts.

tech stack :
fastapi
uvicorn
torch
pandas
numpy
scikit-learn
cryptography
python-dotenv
requests
streamlit
matplotlib