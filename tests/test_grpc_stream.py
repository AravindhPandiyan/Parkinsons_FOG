import dask.dataframe as dd

from grpc_stream import GRPCConnect, GRPCServe, PredictorJob


def data_streamer():
    """
    data_streamer method is used to test the client side streaming of the gRPC connection.
    """
    data = dd.read_parquet("data/raw/unlabeled")
    data[["AccV", "AccML", "AccAP"]] = data[["AccV", "AccML", "AccAP"]] * 9.80665
    grpc_client = GRPCConnect(data.itertuples(), "localhost:50051", 0)
    grpc_client.connect_to_stream()


def server(inference):
    """
    server method is used to test the server side streaming of the gRPC connection.
    """
    job = PredictorJob(inference)
    grpc_server = GRPCServe(job, "localhost:50051")

    try:
        grpc_server.open_line()

    finally:
        grpc_server.close_line()
