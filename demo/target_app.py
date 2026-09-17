"""Small API target adapted from FastAPI's official body update example.

Source project: https://github.com/fastapi/fastapi
Example: docs_src/body_updates/tutorial001_py310.py
"""

from copy import deepcopy

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class Item(BaseModel):
    name: str
    price: float
    tax: float = 10.5


INITIAL_ITEMS = {
    "foo": {"name": "Foo", "price": 50.2, "tax": 10.5},
    "bar": {"name": "Bar", "price": 62.0, "tax": 20.2},
}


def create_app() -> FastAPI:
    """Give each agent scenario its own in-memory data."""
    app = FastAPI(title="AI API Test Demo Target")
    items = deepcopy(INITIAL_ITEMS)
    app.state.items = items

    @app.get("/items/{item_id}", response_model=Item)
    async def read_item(item_id: str) -> Item:
        item = items.get(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Item not found")
        return item

    @app.put("/items/{item_id}", response_model=Item)
    async def update_item(item_id: str, item: Item) -> Item:
        items[item_id] = item.model_dump()
        return items[item_id]

    return app


app = create_app()
items = app.state.items
