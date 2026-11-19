from hospital.local_model import get_model
from hospital.data_loader import load_data
from hospital.trainer import train

X, y = load_data("datasets/hospital_0.csv")

model = get_model()
model = train(model, X, y)