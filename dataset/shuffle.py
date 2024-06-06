import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from queue import Queue
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils
from functools import partial

CHUNKS = 128


#  Having just discovered WebDataset (https://github.com/webdataset/webdataset) and tarp (https://github.com/webdataset/tarp), ...
# TODO: replace entire map-reduce approach with WebDataset/ tarp ?
