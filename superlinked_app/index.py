import fsspec
import json
import os

from dotenv import load_dotenv
from superlinked import framework as sl

TEXT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
IMAGE_MODEL_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

load_dotenv("./superlinked_app/.env")

MODAL_TOKEN_ID=os.environ["MODAL_TOKEN_ID"]
MODAL_TOKEN_SECRET=os.environ["MODAL_TOKEN_SECRET"]
MODAL_BATCH_SIZE=int(os.environ["MODAL_BATCH_SIZE"])
UNIQUE_CATEGORIES_PATH="gs://superlinked-benchmarks-external/amazon-products-images/unique-categories.json"

number_space_params = {
    "average_rating": (1.0, 5.0),
    "rating_number": (1.0, 1898759.0),
    "price": (0.0, 1000000.0),
}


with fsspec.open(UNIQUE_CATEGORIES_PATH) as f:
    unique_categories = json.load(f)

class ProductSchema(sl.Schema):
    # ID
    parent_asin: sl.IdField

    # Categorical / text
    main_category: sl.String | None
    title: sl.String | None
    description: sl.String | None
    categories: sl.String | None
    image_url: sl.Blob | None

    # Numeric
    average_rating: sl.Float | None
    rating_number: sl.Float | None
    price: sl.Float | None


product_schema = ProductSchema()

# Numeric spaces
average_rating_space = sl.NumberSpace(
    product_schema.average_rating,
    min_value=number_space_params["average_rating"][0],
    max_value=number_space_params["average_rating"][1],
    mode=sl.Mode.SIMILAR,
)
rating_number_space = sl.NumberSpace(
    product_schema.rating_number,
    min_value=number_space_params["rating_number"][0],
    max_value=number_space_params["rating_number"][1],
    mode=sl.Mode.SIMILAR,
)
price_space = sl.NumberSpace(
    product_schema.price,
    min_value=number_space_params["price"][0],
    max_value=number_space_params["price"][1],
    mode=sl.Mode.SIMILAR,
)

modal_embedding_config = sl.ModalEngineConfig(
    token_id=MODAL_TOKEN_ID,
    token_secret=MODAL_TOKEN_SECRET,
    app_name="external-benchmark-mor",
    batch_size=MODAL_BATCH_SIZE,
)

# Text similarity spaces
description_space = sl.TextSimilaritySpace(
    product_schema.description,
    model=TEXT_MODEL_NAME,
    model_handler=sl.TextModelHandler.MODAL,
    embedding_engine_config=modal_embedding_config,
)
title_space = sl.TextSimilaritySpace(
    product_schema.title,
    model=TEXT_MODEL_NAME,
    model_handler=sl.TextModelHandler.MODAL,
    embedding_engine_config=modal_embedding_config,
)
categories_space = sl.TextSimilaritySpace(
    product_schema.categories,
    model=TEXT_MODEL_NAME,
    model_handler=sl.TextModelHandler.MODAL,
    embedding_engine_config=modal_embedding_config,
)

# Image space
image_space = sl.ImageSpace(
    product_schema.image_url,
    model=IMAGE_MODEL_NAME,
    model_handler=sl.ModelHandler.MODAL,
    embedding_engine_config=modal_embedding_config,
)

# Categorical space
main_category_space = sl.CategoricalSimilaritySpace(
    product_schema.main_category, categories=unique_categories["main_category"]
)

product_index = sl.Index(
    [
        average_rating_space,
        rating_number_space,
        price_space,
        description_space,
        title_space,
        categories_space,
        image_space,
        main_category_space,
    ],
    fields=[
        product_schema.average_rating,
        product_schema.rating_number,
        product_schema.main_category,
    ],
)
