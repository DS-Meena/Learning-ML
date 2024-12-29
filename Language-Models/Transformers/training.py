from model import model
from preprocessing import X, y

history = model.fit(X, y, epochs=50, batch_size=64, validation_split=0.2)