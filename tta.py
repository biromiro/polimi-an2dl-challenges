import os
import tensorflow as tf
from tqdm import tqdm
import numpy as np


class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(
            os.path.join(path, 'SubmissionModel'))

    def predict(self, X):
        #  perform test time augmentation
        predictions = []
        for i in tqdm(range(len(X))):
            img = X[i]
            img = np.expand_dims(img, axis=0)
            predictions.append(self.model.predict(img))
            # 90 degree rotation
            img = np.rot90(img)
            predictions.append(self.model.predict(img))
            # 180 degree rotation
            img = np.rot90(img)
            predictions.append(self.model.predict(img))
            # 270 degree rotation
            img = np.rot90(img)
            predictions.append(self.model.predict(img))
            # flip horizontally
            img = np.fliplr(img)
            predictions.append(self.model.predict(img))
            # flip vertically
            img = np.flipud(img)
            predictions.append(self.model.predict(img))
            # flip horizontally and vertically
            img = np.fliplr(img)
            img = np.flipud(img)
            predictions.append(self.model.predict(img))
            # flip horizontally and rotate 90 degree
            img = np.rot90(img)
            img = np.fliplr(img)
            predictions.append(self.model.predict(img))

        predictions = np.array(predictions)
        predictions = np.mean(predictions, axis=0)
        return predictions
