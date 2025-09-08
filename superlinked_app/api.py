import os

from dotenv import load_dotenv
from superlinked import framework as sl

from superlinked_app.index import product_index, product_schema
from superlinked_app.query import query

load_dotenv("superlinked_app/.env")

redis_url = "CHANGE ME"
redis_port = 0 # CHANGE ME
redis_username = "CHANGE ME"
redis_password = os.environ.get("REDIS_PASSWORD")
DATA_PATH = (
    "gs://superlinked-benchmarks-external/amazon-products-images/benchmark-10M.parquet"
)
CONCURRENT_EMBEDS = int(os.environ["CONCURRENT_EMBEDS"])

vector_database = sl.RedisVectorDatabase(
    host=redis_url,
    port=redis_port,
    username=redis_username,
    password=redis_password,
)

source_product: sl.RestSource = sl.RestSource(schema=product_schema)

## setup the data loader
product_dl_config = sl.DataLoaderConfig(
    DATA_PATH,
    sl.DataFormat.CSV,
    pandas_read_kwargs={"chunksize": CONCURRENT_EMBEDS * int(os.environ["MODAL_BATCH_SIZE"])},
)

product_data_loader_source = sl.DataLoaderSource(product_schema, product_dl_config)
executor = sl.RestExecutor(
    sources=[
        source_product,
        product_data_loader_source,
    ],
    indices=[product_index],
    queries=[
        sl.RestQuery(sl.RestDescriptor("product_query"), query),
    ],
    vector_database=vector_database,
)

sl.SuperlinkedRegistry.register(executor)
