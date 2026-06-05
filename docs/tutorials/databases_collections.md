# Tutorial: Databases and Collections

This tutorial covers the storage objects you create before inserting vectors:
clients, databases, collections, descriptions, existence checks, and safe
destructive operations.

## Setup

```python
import lynse

client = lynse.VectorDBClient(uri="./db-collection-demo")
```

## Create a database

```python
db = client.create_database("catalog")
```

Create-or-recreate is useful for tests:

```python
db = client.create_database("catalog", drop_if_exists=True)
```

`drop_if_exists=True` deletes the existing database first. Do not use it in
production initialization unless that is exactly what you intend.

Open an existing database:

```python
db = client.get_database("catalog")
```

List and drop databases:

```python
print(client.list_databases())
client.drop_database("old_catalog")
```

LynseDB limits the number of databases created through `VectorDBClient` to 64.
Use collections and metadata fields before creating hundreds of databases.

## Create a collection

Collections need a primary vector dimension:

```python
products = db.require_collection(
    "products",
    dim=384,
    description="Product text embeddings",
)
```

`require_collection()` creates the collection if needed and opens it if it
already exists.

Open an existing collection:

```python
products = db.get_collection("products")
```

Drop a collection:

```python
db.drop_collection("products_archive")
```

## Choose collection boundaries

Use one collection when rows share:

- the same primary embedding dimension;
- the same primary metric and index family;
- similar write and backup lifecycle;
- similar access pattern.

Use separate collections when:

- embeddings come from different models with different dimensions;
- a subset needs a very different index;
- data belongs to different lifecycle domains;
- you want to drop, export, restore, or compact it independently.

Use metadata fields for categories, tenants, languages, timestamps, and source
labels when the vector shape and lifecycle are otherwise the same.

## Descriptions

Descriptions help operations and inspection:

```python
db.update_collection_description(
    "products",
    "Product search collection using 384-dimensional text embeddings.",
)

products.update_description("Updated description from collection handle.")
```

Descriptions are visible through collection details:

```python
print(db.show_collections_details())
```

## Inspect databases and collections

```python
print(db.database_exists())
print(db.show_collections())
print(db.show_collections_details())

print(products.exists())
print(products.shape)
print(products.stats())
print(products.index_mode)
```

`show_collections_details()` returns a pandas DataFrame when pandas is
installed; otherwise it returns a list of dictionaries.

## Read-only local inspection

Open an existing local root in read-only mode:

```python
readonly_client = lynse.VectorDBClient(uri="./db-collection-demo", read_only=True)
readonly_db = readonly_client.get_database("catalog")
readonly_products = readonly_db.get_collection("products")

print(readonly_products.shape)
readonly_client.close()
```

Read-only mode is ignored for remote clients because server-side permissions are
controlled by the HTTP service.

## Warm-up

`get_collection(..., warm_up=True)` and `require_collection(..., warm_up=True)`
allow the client to open the collection with warm-up behavior where supported:

```python
products = db.get_collection("products", warm_up=True)
```

Use it for long-lived services that want predictable first-query behavior.

## Context managers

The top-level client supports context-manager cleanup:

```python
with lynse.VectorDBClient(uri="./db-collection-demo") as client:
    db = client.get_database("catalog")
    products = db.get_collection("products")
    print(products.shape)
```

For long-lived web services, create one client per process and reuse it instead
of constructing a new client per request.

## Naming tips

Use simple names such as:

```text
app
catalog
documents
products
chunks
images
```

Avoid encoding too much state into names. Prefer metadata fields for values that
you will filter by, such as tenant, language, source, or environment.
