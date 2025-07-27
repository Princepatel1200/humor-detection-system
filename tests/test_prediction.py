# tests/test_prediction.py

from scripts.predict import predict_humor

def test_prediction():
    assert predict_humor("Why did the chicken cross the road? To get to the other side!") == "Humorous 😄"
    assert predict_humor("Water boils at 100 degrees Celsius.") == "Not Humorous 😐"

if __name__ == "__main__":
    test_prediction()
    print("All tests passed.")
