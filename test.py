import unittest
import joblib
import pandas as pd

class TestModel(unittest.TestCase):
    restored_model = None
    base_model = None
    top_model_complete = 'model.keras'
    samples_path = "samples"

    def setUp(self):

        self.restored_model = joblib.load("model.weights.h5")
    
    def test_sample1(self):
        sample = pd.DataFrame([[4.6,3.1,1.5,0.2]],
                         columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        # Predict the class
        predicted_class = self.restored_model.predict(sample)
        prediction = predicted_class[0]
        print("Predicted class:", predicted_class[0])
        self.assertEqual(prediction,"setosa","Predicted class is wrong")

    def test_sample2(self):
        sample = pd.DataFrame([[4.6,3.1,1.5,0.2]],
                         columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        # Predict the class
        predicted_class = self.restored_model.predict(sample)
        prediction = predicted_class[0]
        print("Predicted class:", predicted_class[0])
        self.assertEqual(prediction,"setosa","Predicted class is wrong")

if __name__ == '__main__':
    unittest.main()    