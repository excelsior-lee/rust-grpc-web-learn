import io
import logging
from concurrent import futures

from PIL import Image
import grpc
# from keras_preprocessing.image.utils import _PIL_INTERPOLATION_METHODS
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import shape
from tensorflow.python.ops.math_ops import to_int64

import infer_pb2_grpc
import infer_pb2
PreProcessResponse = infer_pb2.PreProcessResponse
AfterProcessResponse = infer_pb2.AfterProcessResponse
Pred = infer_pb2.Pred


class Processer(infer_pb2_grpc.ProcessServicer):
    def PreProcess(self, request, context):
        # 根据 bytes 转化为 img
        img = Image.open(io.BytesIO(request.image))
        img = img.convert('RGB')
        # resample = _PIL_INTERPOLATION_METHODS['nearest']
        img = img.resize((600, 600), resample=Image.NEAREST)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # 数据拉成一维的
        return PreProcessResponse(shape=x.shape, data=x.reshape(-1))

    def AfterProcess(self, request, context):
        # 数据还原成二维的
        preds = np.array(request.data).astype(np.float32).reshape(request.shape)
        # 数据转换为人可读的
        preds = decode_predictions(preds, top=3)[0]
        preds = [Pred(name=name, probability=p) for _, name, p in preds]

        return AfterProcessResponse(preds=preds)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    infer_pb2_grpc.add_ProcessServicer_to_server(Processer(), server)
    addr = '0.0.0.0:5001'
    server.add_insecure_port(addr)
    server.start()
    print("Listen on:", '0.0.0.0:5001')
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()