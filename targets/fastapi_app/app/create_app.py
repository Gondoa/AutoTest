# targets/fastapi_app/app.py

from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ── 数据模型 ───────────────────────────────────────────────────────────────────

class Item(BaseModel):
    name:  Optional[str]   = None
    price: Optional[float] = None
    tax:   Optional[float] = None


# ── 工厂函数 ───────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    每次调用返回一个全新的 FastAPI 实例，带独立的内存数据库。

    这是测试隔离的关键：
    每个子 Agent 调用 create_app() 拿到自己的 app，
    互相的 PUT 操作不会影响对方的 GET 结果。
    """
    app = FastAPI()

    # 每个实例有自己的内存数据——注意是在函数内部定义的，不是模块级全局变量
    db: Dict[str, Item] = {
        "foo": Item(name="Foo", price=50.2, tax=10.5),
        "bar": Item(name="Bar", price=62.0, tax=20.2),
    }

    @app.get("/items/{item_id}")
    async def read_item(item_id: str) -> Dict:
        if item_id not in db:
            raise HTTPException(status_code=404, detail="Item not found")
        item = db[item_id]
        return {"item_id": item_id, **item.model_dump()}

    @app.put("/items/{item_id}")
    async def update_item(item_id: str, item: Item) -> Dict:
        if item_id not in db:
            raise HTTPException(status_code=404, detail="Item not found")
        db[item_id] = item
        return {"item_id": item_id, **item.model_dump()}

    return app