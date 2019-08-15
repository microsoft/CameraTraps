# Cosmos database

We would like to centralize all COCO Camera Trap (CCT) format json databases containing image metadata and species/bounding box annotations into an instance of a Cosmos DB for easier management and querying. 

- `make_embedded_db.py` converts a json database in the CTT format to an embedded/denormalized format, so that the image metadata and annotations on image are both within the image entry.
 
    - Be very careful to make sure the `dataset_name` parameter to this script (first positional argument) is set properly - bulk updating entries is very slow. 
    
    - You can then insert these entries in bulk using the [data migration tool](https://docs.microsoft.com/en-us/azure/cosmos-db/import-data) (Windows only). Need to leave both the Partition Key and the ID fields blank, and uncheck "suppress ID generation". 
    
    - Each dataset's metadata stored (each dataset's data is one partition) cannot exceed [10GB](https://docs.microsoft.com/en-us/azure/cosmos-db/concepts-limits).

- `useful_queries.ipynb` is a collection of queries in SQL syntax for common operations such as getting images with a certain species, recently inserted images, and images with bounding box annotations. It also shows how to connect to the database using the Python SDK.

    - Documentation: https://docs.microsoft.com/en-us/azure/cosmos-db/create-sql-api-python


### TODO 

-[ ] Migrate all CCT databases there for datasets that have their images stored unzipped in Blob Storage

-[ ] Create an additional table for dataset metadata


