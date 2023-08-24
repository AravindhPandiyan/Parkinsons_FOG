import time
from collections import deque

import pandas as pd

from grpc_stream import service_pb2
from grpc_stream import service_pb2_grpc


class Jobs:
    pass


class PredictorJob(Jobs, service_pb2_grpc.PackageServicer):
    def __init__(self, inference):
        self.infer = inference
        self.w_size = self.infer.window_size
        self.buffer = deque(maxlen=self.w_size)
        self.steps = self.infer.steps
        self.count = 0

    def bidirectionalStream(self, request_iterator, context):
        for package in request_iterator:
            data = [package.AccV, package.AccML, package.AccAP]
            if self.count < self.steps:
                self.buffer.append(data)
                self.count += 1

            elif len(self.buffer) != self.w_size:
                self.buffer.append(data)
                self.count += 1

            else:
                df = pd.DataFrame(self.buffer)
                pred = self.infer.predict_fog(df)
                pred = pred.astype(int)
                pred = pred[0].tolist()
                pred = service_pb2.Prediction(StartHesitation=pred[0], Turn=pred[1], Walking=pred[2])
                yield pred
                time.sleep(1)
                self.buffer.append(data)
                self.count = 1
