import time
from collections import deque

import pandas as pd

from grpc_stream import rpc_service_pb2, rpc_service_pb2_grpc
from logger_config import logger as log


class Jobs:
    """
    `Jobs` is an empty class used for **grouping classes** of a similar type under one. This class is **inherited**
    by the similar classes to become part of the same **group**. The work of any **'JOB'** class is for setting
    up services like **Unary RPC, Server-side streaming RPC, Client-side streaming RPC and Bi-directional streaming
    RPC**. The 'JOB' classes will be specific to their work. There can exist multiple **'JOB'** of same **service**.
    """

    NAME = "JOB"


class PredictorJob(Jobs, rpc_service_pb2_grpc.PackageServicer):
    """
    `PredictorJob` is a **service** for the servicing the **client connections**.
    """

    CHILD_NAME = f"PREDICTOR_{Jobs.NAME}"

    def __init__(self, inference):
        """
        `PredictorJob` class is part of the main **'JOB'** group of classes. This **'JOB'** is used for
        **Bi-directional streaming** of data and predictions between **client** and **server** respectively. This
        class also **inherits the Servicer** class from `pb2_grpc.py` file.

        Params:
            `inference`: inference is expecting the inference class of the model.
        """

        self.infer = inference
        self.w_size = self.infer.window_size
        self.buffer = deque(maxlen=self.w_size)
        self.steps = self.infer.steps
        self.count = 0

    def bidirectionalStream(self, request_iterator, context):
        """
        `bidirectionalStream` method is used to **override** the method in the servicer class and allows to
        write down the specific code for handling **Bi-directional streaming** of the data packages.

        Params:
            `request_iterator`: request_iterator is used for **iterating** over the data packets sent form
            the client side.

            `context`: context contains the **metadata** of the **client**, method name and deadline for the calls**.

        Returns:
            This method **streams predictions to the client**. It acts like a **Generator**.
        """
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
                pred = rpc_service_pb2.Prediction(
                    StartHesitation=pred[0], Turn=pred[1], Walking=pred[2]
                )
                yield pred
                time.sleep(1)
                self.buffer.append(data)
                self.count = 1
