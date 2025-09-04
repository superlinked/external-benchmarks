from superlinked import framework as sl

from superlinked_app.index import product_index, product_schema

query = (
    sl.Query(product_index)
    .find(product_schema)
    .with_vector(product_schema, sl.Param("product_id"), 1.0)
    .filter(product_schema.average_rating <= sl.Param("rating_max"))
    .filter(product_schema.rating_number >= sl.Param("rating_num_min"))
    .filter(product_schema.main_category == sl.Param("main_category"))
    .limit(100)
)
