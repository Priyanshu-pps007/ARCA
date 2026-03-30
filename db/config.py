from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI
import os
import asyncio
from model import *

load_dotenv()

postgres_password = os.getenv("POSTGRES_PASSWORD")
database_url = f"postgresql+asyncpg://postgres:{postgres_password}@localhost:5432/ARCA_db"

engine = create_async_engine(database_url, echo=True)

# Session factory — produces one AsyncSession per request
SessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False   # ← important: keeps ORM objects usable after commit
)

# FastAPI dependency — inject this into every route that needs DB access
async def get_session():
    async with SessionFactory() as session:
        yield session

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield
    # shutdown
    await engine.dispose()

async def connect_db():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

if __name__ == '__main__':
    asyncio.run(connect_db())