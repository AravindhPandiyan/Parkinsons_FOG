import dask.dataframe as dd

from grpc_stream import GRPCConnect
from grpc_stream import PredictorJob, GRPCServe


def data_streamer():
    data = dd.read_parquet('data/raw/unlabeled')
    data[['AccV', 'AccML', 'AccAP']] = data[['AccV', 'AccML', 'AccAP']] * 9.80665
    grpc_client = GRPCConnect("localhost:50051", data.itertuples())
    grpc_client.connect_to_stream()


def server(inference):
    job = PredictorJob(inference)
    grpc_server = GRPCServe(job, "localhost:50051")
    grpc_server.open_line()
