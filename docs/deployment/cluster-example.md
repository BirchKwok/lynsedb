# Lightweight Cluster Example

Start ordinary LynseDB servers as shards, then point the coordinator at them.
The coordinator owns metadata, mirrors writes to active replicas, and only
auto-promotes replicas that have not missed a write.

Shard nodes also start an internal custom RPC listener automatically. You do not
need to configure it: the coordinator derives the RPC port from the shard HTTP
port (`http_port + 10000`, or `http_port - 10000` for high ports) and falls back
to HTTP if RPC is unavailable. For example, shard HTTP `7638` uses internal RPC
`17638`. Vector search, batch search, delete/restore, and add/upsert writes use
RPC when available. For writes with fields, vectors are sent as raw `float32`
bytes and fields are appended as an optional compact binary payload segment,
preserving current field semantics without JSON serialization on the cluster
hot path.

```json
{
  "bucket_count": 4096,
  "write_mirror_replicas": true,
  "shard_groups": [
    {
      "name": "sg0",
      "primary": "http://127.0.0.1:7638",
      "replicas": [
        "http://127.0.0.1:7639"
      ]
    },
    {
      "name": "sg1",
      "primary": "http://127.0.0.1:7640",
      "replicas": [
        "http://127.0.0.1:7641"
      ]
    }
  ]
}
```

```bash
lynse serve --port 7638 --data-dir ./data/sg0-primary
lynse serve --port 7639 --data-dir ./data/sg0-replica
lynse serve --port 7640 --data-dir ./data/sg1-primary
lynse serve --port 7641 --data-dir ./data/sg1-replica
lynse serve --role coordinator --port 7637 \
  --cluster-config ./cluster.json \
  --cluster-state ./cluster_state.json
```

Clients continue to use the coordinator as a normal LynseDB HTTP endpoint:

```python
from lynse import VectorDBClient

db = VectorDBClient("http://127.0.0.1:7637").create_database("demo")
col = db.require_collection("docs", dim=768)
```
