from RAG.ingester import Retriever
import asyncio

retrieve = Retriever()

async def main():
    data = await retrieve.retrieve("fastapi middleware")
    print(data)

if __name__ == "__main__":
    asyncio.run(main())