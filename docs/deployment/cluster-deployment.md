# Cluster Deployment and Maintenance

LynseDB cluster mode is a lightweight sharding layer for remote HTTP
deployments. You start several ordinary LynseDB servers as shard nodes, then
start one coordinator in front of them. Applications keep using
`VectorDBClient("http://coordinator:7637")` just like a single remote server.

Use cluster mode when one server is not enough for your data size or query
throughput, or when you want a primary plus replica layout for shard failover.
For a first production deployment, start with one coordinator and at least two
shard groups. Each shard group should have one primary and, if you need
failover, one replica.

## Architecture

```text
Python client
    |
    v
Coordinator :7637
    |
    +-- shard group sg0
    |     +-- primary http://10.0.0.11:7638
    |     +-- replica http://10.0.0.12:7638
    |
    +-- shard group sg1
          +-- primary http://10.0.0.21:7638
          +-- replica http://10.0.0.22:7638
```

The coordinator owns cluster metadata and request routing:

- collection metadata is stored in the coordinator state file;
- IDs are mapped to shard groups with stable hash buckets;
- writes are routed to the owning shard group;
- searches fan out to all shard groups and are merged by the coordinator;
- writes are mirrored to active replicas when `write_mirror_replicas` is true;
- a replica that misses a write is marked `stale` and is not auto-promoted.

Shard nodes are normal LynseDB HTTP servers. They also start an internal custom
RPC listener automatically. The coordinator derives the internal RPC port from
the shard HTTP port: `http_port + 10000`, or `http_port - 10000` for high
ports. For example, shard HTTP port `7638` uses internal RPC port `17638`.
Search, batch search, add, upsert, delete, and restore use RPC when available
and fall back to HTTP when RPC is unavailable.

## Before You Start

Install LynseDB on every node:

```shell
pip install lynsedb
```

Plan these pieces before production:

| Item | Recommendation |
| --- | --- |
| Coordinator | Run one active coordinator. Keep its state file on persistent storage. |
| Shards | Run one LynseDB server per primary or replica data directory. |
| Data directories | Give every shard process its own persistent directory. Do not share one directory between nodes. |
| Network | Let clients reach the coordinator. Let the coordinator reach every shard HTTP port and derived RPC port. |
| Auth | Use `--api-key` on shards and `--shard-api-key` on the coordinator. Protect the coordinator with a private network or reverse proxy. |
| Backups | Back up every shard data directory and the coordinator state file together. |

The coordinator currently does not enforce client authentication itself. If the
coordinator is reachable outside a trusted network, put it behind a reverse
proxy, gateway, firewall, or service mesh that provides authentication.

## Quick Local Cluster

This example starts two shard groups on one machine. It is the easiest way to
learn the moving parts before deploying on separate hosts.

Create a cluster config:

```json title="cluster.json"
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

Start the shard nodes in separate terminals:

```shell
lynse serve --host 127.0.0.1 --port 7638 --data-dir ./data/sg0-primary
lynse serve --host 127.0.0.1 --port 7639 --data-dir ./data/sg0-replica
lynse serve --host 127.0.0.1 --port 7640 --data-dir ./data/sg1-primary
lynse serve --host 127.0.0.1 --port 7641 --data-dir ./data/sg1-replica
```

Start the coordinator:

```shell
lynse serve \
  --role coordinator \
  --host 127.0.0.1 \
  --port 7637 \
  --cluster-config ./cluster.json \
  --cluster-state ./cluster_state.json
```

Check that the coordinator is running:

```shell
curl http://127.0.0.1:7637/
curl http://127.0.0.1:7637/cluster_info
```

Use the coordinator from Python:

```python
from lynse import VectorDBClient

client = VectorDBClient("http://127.0.0.1:7637")
db = client.create_database("demo", drop_if_exists=True)
collection = db.require_collection("docs", dim=4)

collection.add(
    ids=["doc-1", "doc-2"],
    vectors=[
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
    ],
    fields=[
        {"title": "first"},
        {"title": "second"},
    ],
)
collection.commit()

result = collection.search([0.1, 0.2, 0.3, 0.4], k=2, return_fields=True)
print(result)
```

## Production Layout

In production, run each shard primary and replica on separate machines or
separate failure domains.

Example:

```json title="/etc/lynsedb/cluster.json"
{
  "bucket_count": 4096,
  "write_mirror_replicas": true,
  "shard_groups": [
    {
      "name": "sg0",
      "primary": "http://10.0.0.11:7638",
      "replicas": [
        "http://10.0.0.12:7638"
      ]
    },
    {
      "name": "sg1",
      "primary": "http://10.0.0.21:7638",
      "replicas": [
        "http://10.0.0.22:7638"
      ]
    }
  ]
}
```

Start each shard:

```shell
lynse serve \
  --host 0.0.0.0 \
  --port 7638 \
  --data-dir /var/lib/lynsedb/shard \
  --api-key "${LYNSE_SHARD_API_KEY}"
```

Start the coordinator:

```shell
lynse serve \
  --role coordinator \
  --host 0.0.0.0 \
  --port 7637 \
  --cluster-config /etc/lynsedb/cluster.json \
  --cluster-state /var/lib/lynsedb/coordinator/cluster_state.json \
  --shard-api-key "${LYNSE_SHARD_API_KEY}" \
  --health-interval-secs 1.0 \
  --health-failures 3
```

On later coordinator restarts, keep using the same `--cluster-state` file. You
may pass both `--cluster-config` and `--cluster-state`, but the existing state
file is the source of truth after the first start.

## Configuration Reference

Cluster config fields:

| Field | Default | Description |
| --- | --- | --- |
| `bucket_count` | `4096` | Number of routing buckets assigned to shard groups when a collection is created. Keep this value stable after data exists. |
| `write_mirror_replicas` | `true` | Mirror writes to replicas whose state is `active`. |
| `shard_groups` | required | List of shard groups. Each group needs a `primary` URI and may have `replicas`. |
| `state_path` | `cluster_state.json` | Optional coordinator state path when `--cluster-state` is not provided. |
| `shard_api_key` | none | Optional key used by the coordinator when forwarding requests to shards. |

Replica entries can be simple strings:

```json
"replicas": ["http://10.0.0.12:7638"]
```

Or objects with an explicit state:

```json
"replicas": [
  {"uri": "http://10.0.0.12:7638", "state": "active"}
]
```

Coordinator CLI flags:

| Flag | Description |
| --- | --- |
| `--role coordinator` | Start coordinator mode instead of a normal shard server. |
| `--cluster-config` | JSON config used to seed the state file on first start. |
| `--cluster-state` | Mutable coordinator metadata file. Back this up. |
| `--shard-api-key` | API key sent to shard nodes as `Authorization: Bearer ...`. |
| `--request-timeout-secs` | Timeout for coordinator-to-shard requests. |
| `--health-interval-secs` | Interval between shard health probes. |
| `--health-failures` | Consecutive failed probes before a node is considered unhealthy. |

Environment variables are also supported:

```shell
export LYNSE_ROLE=coordinator
export LYNSE_CLUSTER_CONFIG=/etc/lynsedb/cluster.json
export LYNSE_CLUSTER_STATE=/var/lib/lynsedb/coordinator/cluster_state.json
export LYNSE_SHARD_API_KEY=your_shard_key
export LYNSE_HEALTH_INTERVAL_SECS=1.0
export LYNSE_HEALTH_FAILURES=3
```

## How Routing Works

When a collection is created, the coordinator stores its routing table in
`cluster_state.json`. Each bucket points to one shard group.

For explicit string or integer IDs, LynseDB hashes the external ID and routes
the item to the owning shard group. For records that need generated integer
IDs, the coordinator allocates IDs and then routes them. This means clients can
continue to use normal `add`, `upsert`, `delete`, `restore`, `search`, and
query APIs through the coordinator.

Search requests are sent to every shard group. The coordinator asks one healthy
node per group, merges the per-shard top results, and returns one result set to
the client.

## Health and Failover

The coordinator probes every primary and replica. A node is marked unhealthy
after `--health-failures` consecutive failed probes.

If a primary is unhealthy and an active healthy replica exists in the same shard
group, the coordinator promotes that replica. The old primary is moved into the
replica list with state `stale`.

Check cluster state:

```shell
curl http://127.0.0.1:7637/cluster_info
```

Look for:

- `primary`: current primary URI for each shard group;
- `primary_epoch`: increments when a new primary is promoted;
- replica `state`: `active` replicas can receive mirrored writes and can be
  promoted; `stale` replicas cannot be promoted automatically;
- `meta_epoch`: increments when coordinator metadata changes.

Important behavior:

- If a primary write fails, the client request fails.
- If a replica write fails, the request can still succeed, but that replica is
  marked `stale`.
- A stale replica may be missing data and must be rebuilt before it is marked
  active again.
- If a shard group has no healthy active node, reads and writes for that group
  will fail until a node is restored.

## Maintenance Tasks

### Restart a Shard

For a short restart of a healthy node:

1. Stop the shard process.
2. Start it again with the same `--data-dir`, host, port, and API key.
3. Check `curl http://shard-host:7638/healthz`.
4. Check `curl http://coordinator:7637/cluster_info`.

If the node did not miss writes, it can continue serving. If it missed writes
and is marked `stale`, rebuild it before relying on it.

### Restart the Coordinator

The coordinator is stateless except for its state file. Restart it with the same
`--cluster-state` path:

```shell
lynse serve \
  --role coordinator \
  --host 0.0.0.0 \
  --port 7637 \
  --cluster-state /var/lib/lynsedb/coordinator/cluster_state.json \
  --shard-api-key "${LYNSE_SHARD_API_KEY}"
```

Pass `--cluster-config` again only if you want it available for validation or
for first-start convenience. Existing state is loaded from `--cluster-state`.

### Rebuild a Stale Replica

A stale replica should be treated as out of date. Rebuild it from the current
primary for that shard group.

Recommended process:

1. Stop the stale replica.
2. Keep it out of service while rebuilding.
3. Take a consistent backup or filesystem copy from the current primary data
   directory during a maintenance window.
4. Restore that copy into the replica data directory.
5. Start the replica with the same port and API key.
6. Verify the replica health endpoint.
7. Update `cluster_state.json` and change that replica state from `stale` to
   `active`.
8. Restart the coordinator so it reloads the edited state file.

Only mark a replica `active` after you are confident it has the same data as the
current primary.

### Add Shards

New shard groups are used by collections created after the routing table
includes them. Existing collections keep their original `bucket_to_group`
mapping and are not automatically rebalanced.

For a new deployment, choose the shard group count before loading large data.
For an existing deployment, the safest scaling path is:

1. Start a new cluster config with the desired shard groups.
2. Create a new database or collection.
3. Re-ingest or migrate the data through the coordinator.
4. Switch application traffic after validation.

Do not edit `bucket_to_group` for an existing collection unless you are also
moving the corresponding data and have tested the migration offline.

### Back Up a Cluster

Back up these items together:

- every shard primary data directory;
- every replica data directory if you want faster replica restore;
- the coordinator `cluster_state.json`;
- the cluster config and service files.

Before a planned backup, run checkpoints through the coordinator:

```python
from lynse import VectorDBClient

client = VectorDBClient("http://coordinator:7637")
db = client.get_database("demo")
collection = db.get_collection("docs")
collection.checkpoint()
```

For multiple collections, checkpoint each collection. Then snapshot or copy the
coordinator state file and shard data directories as one backup set.

### Upgrade

Use a maintenance window for upgrades:

1. Back up all shard data directories and `cluster_state.json`.
2. Stop writes at the application layer.
3. Checkpoint active collections.
4. Upgrade replicas first and start them.
5. Upgrade primaries.
6. Restart the coordinator.
7. Run smoke tests through the coordinator.
8. Resume writes.

For large clusters, test the exact version upgrade on a staging copy first.

## Monitoring

Monitor each shard as a normal LynseDB server:

```shell
curl http://10.0.0.11:7638/healthz
curl http://10.0.0.11:7638/readyz
curl http://10.0.0.11:7638/metrics
```

Monitor the coordinator:

```shell
curl http://coordinator:7637/
curl http://coordinator:7637/cluster_info
```

Alert on:

- shard health or readiness failures;
- coordinator process down;
- replica state changing to `stale`;
- unexpected `primary_epoch` changes;
- disk usage on shard data volumes;
- high query latency or write failures on shards.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Coordinator will not start | On first start, pass `--cluster-config`. On later starts, pass an existing `--cluster-state`. |
| `cluster config requires at least one shard group` | The config must contain `shard_groups` or `shards` with at least one entry. |
| Shard requests return unauthorized | Make sure shards use `--api-key` and the coordinator uses the same `--shard-api-key`. |
| RPC is not used | Open the derived RPC port between coordinator and shard, or let the coordinator fall back to HTTP. |
| Replica becomes `stale` | It missed a mirrored write. Rebuild it from the primary before marking it active. |
| Search misses recent writes | Confirm writes were sent to the coordinator, check shard health, then inspect `cluster_info` for stale or promoted nodes. |
| Existing data does not rebalance after adding a shard | Existing collection routing is fixed. Create a new collection and migrate or re-ingest. |

## Operational Checklist

Before accepting production traffic:

- shard nodes have persistent data directories;
- the coordinator state file is on persistent storage;
- coordinator-to-shard HTTP and derived RPC ports are reachable;
- shard authentication is configured if the shard network is not fully trusted;
- coordinator access is protected by network policy or a proxy;
- `/cluster_info` shows the expected primaries and active replicas;
- a backup and restore procedure has been tested;
- application clients connect only to the coordinator.
