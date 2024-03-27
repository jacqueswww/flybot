import logging

from data_collections.models import DataCollection
from sanic import Sanic, response

from tortoise.contrib.sanic import register_tortoise

logging.basicConfig(level=logging.DEBUG)

app = Sanic(__name__)


@app.get("/data_collections")
async def list_all(request):
    collections = await DataCollection.all()
    return response.json({
        "success": True,
        "results": [str(dc) for dc in collections]
    })


@app.post("/data_collections")
async def create_(request):
    dc = await DataCollection.create(
        name="New Collection",
        collection_type='zendesk',
    )
    return response.json({
        "success": True,
        "results": str(dc)
    })


DB_URL = "sqlite://./local_db.sqlite"
DB_APPS = {
    "data_collections": {
        "models": ["aerich.models", "data_collections.models"],
        "default_connection": "default",
    },
}


register_tortoise(
    app,
    db_url=DB_URL,
    modules={'data_collections': ["aerich.models", "data_collections.models"]},
    generate_schemas=True
)


TORTOISE_ORM = {
    "connections": {
        "default": DB_URL
    },
    "apps": DB_APPS
}


if __name__ == "__main__":
    app.run(port=5000)
