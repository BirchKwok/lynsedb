[MinVectorDB](https://github.com/BirchKwok/MinVectorDB/blob/main/MinVecDB.ipynb) is a simple vector storage and query database implementation, providing clear and concise Python APIs aimed at lowering the barrier to using vector databases. More practical features will be added in the future. 

It is important to note that MinVectorDB is not designed for efficiency and thus does not include built-in algorithms like approximate nearest neighbors for efficient searching. 

It originated from the author's need to demonstrate a large language model demo, designed for 100% recall. 

Additionally, it has not undergone rigorous code testing, so caution is advised when using it in a production environment.

[MinVectorDB](https://github.com/BirchKwok/MinVectorDB/blob/main/MinVecDB.ipynb) 是简易实现的向量存储和查询数据库，提供简洁明了的python API，旨在降低向量数据库的使用门槛。未来将添加更多实用功能。

需要注意的是，MinVectorDB并非为追求效率而生，因此，并没有内置近似最近邻等高效查找算法。

它起源于作者需要演示大语言模型Demo的契机，为了追求100%召回率而设计，此外，也没有经过严格的代码测试，因此如果将其用于生产环境需要特别谨慎。

# Install
```shell
pip install MinVectorDB
```


# Quick start

## Demo 1-2

```python
from IPython.display import display_markdown

import numpy as np

from spinesUtils.utils import Timer
from min_vec import MinVectorDB

timer = Timer()

# ===================================================================
# ========================= DEMO 1 ==================================
# ===================================================================
# Demo 1 -- Sequentially add vectors.
# Create a MinVectorDB instance.
display_markdown("*Demo 1* -- **Sequentially add vectors**", raw=True)

timer.start()
db = MinVectorDB(dim=1024, database_path='test.mvdb', chunk_size=100)

np.random.seed(23)

# Define the initial ID.
id = 0
for t in np.random.random((1000, 1024)):
    # Vectors need to be normalized before writing to the database.
    t = t / np.linalg.norm(t)
    db.add_item(t, id=id)

    # ID increments by 1 with each loop iteration.
    id += 1
db.commit()
print(f"\n* [Insert data] Time cost {timer.last_timestamp_diff():>.4f} s.")
timer.middle_point()

res = db.query(db.head(10)[0], k=10)
print("  - Query vector: ", db.head(10)[0])
print("  - Database index of top 10 results: ", res[0])
print("  - Cosine similarity of top 10 results: ", res[1])
print(f"\n* [Query data] Time cost {timer.last_timestamp_diff():>.4f} s.")
timer.middle_point()

# For demonstrating Demo2, the database created in Demo1 needs to be deleted, but this operation is not required in actual use.
db.delete()

del db

display_markdown("------", raw=True)

# ===================================================================
# ========================= DEMO 2 ==================================
# ===================================================================
# Demo 2 -- Bulk add vectors.
display_markdown("*Demo 2* -- **Bulk add vectors**", raw=True)
# print("# This is the demonstration area for Demo 2 -- Bulk add vectors.")

timer.middle_point()

db = MinVectorDB(dim=1024, database_path='test.mvdb', chunk_size=100)

np.random.seed(23)

# Define the initial ID.
id = 0
vectors = []
for t in np.random.random((1000, 1024)):
    # Vectors need to be normalized before writing to the database.
    t = t / np.linalg.norm(t)
    vectors.append((t, id))
    # ID increments by 1 with each loop iteration.
    id += 1

db.bulk_add_items(vectors)
db.commit()

print(f"\n* [Insert data] Time cost {timer.last_timestamp_diff():>.4f} s.")
timer.middle_point()

res = db.query(db.head(10)[0], k=10)
print("  - Query vector: ", db.head(10)[0])
print("  - Database index of top 10 results: ", res[0])
print("  - Cosine similarity of top 10 results: ", res[1])
print(f"\n* [Query data] Time cost {timer.last_timestamp_diff():>.4f} s.")

timer.end()
# This operation is not required in actual use.
db.delete()
```

## Demo 3

```python
import numpy as np
from IPython.display import display_markdown

from spinesUtils.utils import Timer
from min_vec import MinVectorDB

timer = Timer()

# ===================================================================
# ========================= DEMO 3 ==================================
# ===================================================================
# Demo 3 -- Use field to improve Searching Recall
display_markdown("*Demo 3* -- **Use field to improve Searching Recall**", raw=True)

timer.start()

db = MinVectorDB(dim=1024, database_path='test.mvdb', chunk_size=100)

np.random.seed(23)

# Define the initial ID.
id = 0
vectors = []
for t in np.random.random((1000, 1024)):
    # Vectors need to be normalized before writing to the database.
    t = t / np.linalg.norm(t)
    vectors.append((t, id, 'test_' + str(id // 100)))
    # ID increments by 1 with each loop iteration.
    id += 1

db.bulk_add_items(vectors)
db.commit()

print(f"\n* [Insert data] Time cost {timer.last_timestamp_diff():>.4f} s.")
timer.middle_point()

res = db.query(db.head(10)[0], k=10, field=['test_0', 'test_3'])
print("  - Query vector: ", db.head(10)[0])
print("  - Database index of top 10 results: ", res[0])
print("  - Cosine similarity of top 10 results: ", res[1])
print(f"\n* [Query data] Time cost {timer.last_timestamp_diff():>.4f} s.")

timer.end()
db.delete()
```

