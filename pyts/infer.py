import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import (decode_predictions, preprocess_input)

from tensorflow.keras.preprocessing import image

# 加载模型
loaded = tf.saved_model.load('pyts/resnet50')
infer = loaded.signatures['serving_default']

# 准备推理数据
img = image.load_img('pyts/test.png', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 进行推理
preds = infer(tf.constant(x))['predictions'].numpy()


# print(preds)
print(decode_predictions(preds, top=3)[0])

print('X: ', x.shape, x.dtype)
with open('pyts/request', 'wb') as f:
    f.write(x.tobytes())