# Importing published attack graphs

`rlattack import` converts an externally published attack graph into a validated,
anonymized scenario. Results measured only on this repository's generator describe that
generator; importing lets the same agents and metrics run on structures someone else
published.

```bash
rlattack import --input topology.graphml --output artifacts/imported.json
rlattack demo --help   # the imported scenario is a normal scenario file
```

## What the importer guarantees

- **No identifiers survive.** Every node is renamed `host-00`, `host-01`, … The mapping
  is not written anywhere, so an imported scenario cannot be traced back to the source
  topology by reading it.
- **Unused attributes are never copied.** The importer reads only `kind`/`type` on
  nodes and `cost`/`weight` on edges. Anything else in the file - descriptions, fact
  strings, owner tags - cannot reach the scenario, because nothing reads it.
- **Obvious live data is refused rather than silently anonymized.** The payload must
  pass the same check as the ThreatGraph adapter: a `hostname`, `ip`, `url`, `password`,
  `token`, `payload`, or `exploit` field, or any string containing a URL, an IP address,
  or a name of three or more dotted labels, rejects the import outright. This is a
  heuristic, not a proof: a two-label name in an unexpected field would pass the check,
  and is then discarded by the previous guarantee rather than caught by this one.
- **Structure is preserved.** Node count, edge count, out-degrees, and edge costs are
  carried over unchanged.

## The import contract

The importer reads a directed graph and treats its nodes as hosts and its edges as
reachability. That is deliberately the smallest contract that most published attack
graphs satisfy.

| Element | Read from | Default |
| --- | --- | --- |
| Host node | any node whose `kind`/`type` is `host`, `node`, `machine`, `asset`, or absent | — |
| Reachability edge | every edge between two host nodes | — |
| Edge cost | edge attribute `cost`, else `weight` | `1.0` |
| Entry host | a node with in-degree 0, else the first host | — |
| Objective host | the reachable host furthest from the entry | — |

Nodes of any other `kind` are ignored rather than rejected, so a graph that also
publishes services or vulnerabilities imports its host layer cleanly.

## Formats

| Extension | Reader |
| --- | --- |
| `.graphml` | `networkx.read_graphml` |
| `.gml` | `networkx.read_gml` |
| `.json` | `networkx.node_link_graph` (node-link JSON) |

## Mapping a specific export

Most tools do not emit the contract directly. Convert once with NetworkX, then import.

### MulVAL-style attack graphs

MulVAL's `AttackGraph` output distinguishes LEAF, AND, and OR nodes, and its edges are
derivation steps rather than reachability. Project the host-to-host relation first:

```python
import networkx as nx

source = nx.read_graphml("AttackGraph.graphml")
hosts = nx.DiGraph()
for node, data in source.nodes(data=True):
    if data.get("type") == "LEAF" and "hostname" in data.get("fact", ""):
        hosts.add_node(node, kind="host")
# add an edge wherever a derivation chain connects two host facts
nx.write_graphml(hosts, "topology.graphml")
```

`fact` strings carry hostnames. A fully qualified one (`webserver.corp.example`) is
refused outright; a short one is ignored rather than imported, since the importer never
reads that attribute. Strip them anyway - a file you did not intend to share should not
be the thing you hand to a tool.

### CyberBattleSim topologies

`CyberBattleSim` environments expose `environment.network`, already a NetworkX graph of
nodes and connections:

```python
import networkx as nx

graph = nx.DiGraph()
graph.add_edges_from(environment.network.edges())
nx.write_graphml(graph, "topology.graphml")
```

Node IDs there are descriptive strings (`"Website"`, `"AzureVM"`); they are replaced on
import, so nothing needs stripping unless a node attribute carries an address.

### Plain adjacency data

A CSV or edge list needs no tool:

```python
import networkx as nx

graph = nx.read_edgelist("edges.txt", create_using=nx.DiGraph)
nx.write_graphml(graph, "topology.graphml")
```

## Topology-only imports

By default the importer attaches a deterministic service, vulnerability, credential, and
objective layer, because a published graph that carries only reachability would import
as a scenario with nothing to do. Pass `--topology-only` to keep the structure alone -
useful when you intend to attach your own layer, and honest when you want to state that
only the topology came from the source.

```bash
rlattack import --input topology.graphml --topology-only --output artifacts/topology.json
```

## What an imported scenario does not tell you

The synthesized exploitation layer is this simulator's, not the source's. An agent
scoring well on an imported topology has handled that topology's *shape* - its
branching, depth, and shortcuts - and nothing about the vulnerabilities the source
described. Say so when reporting such a result.
