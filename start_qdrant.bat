mkdir qdrant_storage
docker run -d --name qdrant -p 6333:6333 -v %cd%\qdrant_storage:/qdrant/storage:z qdrant/qdrant