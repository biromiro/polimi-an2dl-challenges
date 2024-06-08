import os
import tensorflow as tf
import cv2
import numpy as np


class model:
    def __init__(self, path):
        self.model1 = tf.keras.models.load_model(
            os.path.join(path, 'Submission1'), compile=False)
        self.model2 = tf.keras.models.load_model(
            os.path.join(path, 'Submission2'), compile=False)
        self.model3 = tf.keras.models.load_model(
            os.path.join(path, 'Submission3'), compile=False)

    def predict(self, X):

        # predict ensemble
        yhat1 = self.predict_tta(self.model1, X)
        yhat2 = self.predict_tta(self.model2, X)
        yhat3 = self.predict_tta(self.model3, X)

        # argmax across classes
        out = tf.argmax((yhat1 + yhat2 + yhat3) / 3, axis=-1)

        return out

    def rotate_image(self, image, angle):  # and flip
        image = cv2.flip(image, 1)
        rows, cols, _ = image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, M, (cols, rows))
        return rotated_image

    def translate_image(self, image, tx, ty):  # and flip
        image = cv2.flip(image, 0)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(
            image, M, (image.shape[1], image.shape[0]))
        return translated_image

    def zoom_image(self, image, factor):
        rows, cols, _ = image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, factor)
        zoomed_image = cv2.warpAffine(image, M, (cols, rows))
        return zoomed_image

    def flip_image(self, image, flip_horizontal=False, flip_vertical=False):
        if flip_horizontal:
            image = cv2.flip(image, 1)
        if flip_vertical:
            image = cv2.flip(image, 0)
        return image

    def augment_image(self, image):
        # Select the parameters of yours augmentations
        rotated_image = self.rotate_image(np.array(image), 5)
        translated_image = self.translate_image(np.array(image), 3, 3)
        zoomed_image = self.zoom_image(np.array(image), 1.1)
        flipped_image1 = self.flip_image(
            np.array(image), flip_horizontal=True, flip_vertical=False)
        flipped_rotate = self.rotate_image(self.flip_image(
            np.array(image), flip_horizontal=False, flip_vertical=True), -6)

        return image, rotated_image, translated_image, zoomed_image, flipped_image1, flipped_rotate

    # note: it returns predictions like model.predic -> [prob Class1, probClass2]
    def predict_tta(self, model, images):
        predictions_aggregate = []
        for img in images:
            aug_images = np.array(self.augment_image(img))
            predictions = model.predict(aug_images, verbose=0)
            pred_agg = np.mean(predictions, axis=0)
            # pred = np.argmax(pred_agg, axis=1)
            predictions_aggregate.append(pred_agg)
        return tf.convert_to_tensor(np.array(predictions_aggregate))
