"""Graph store abstraction over Neo4j, with an in-memory fake implementation
used for tests and zero-dependency demos.

Every adapter implements the same :class:`GraphStore` interface so the RAG
pipeline and the decay engine never need to know which backend is active.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from app.schemas import GraphEdge, MemoryNode, NodeStatus
from app.tenancy import neo4j_database_name


@dataclass
class HopResult:
    node: MemoryNode
    hops: int
    relation_path: list[str]


class GraphStore(ABC):
    @abstractmethod
    def upsert_node(self, node: MemoryNode) -> None: ...

    @abstractmethod
    def get_node(self, tenant_id: str, node_id: str) -> MemoryNode | None: ...

    @abstractmethod
    def upsert_edge(self, edge: GraphEdge) -> None: ...

    @abstractmethod
    def record_access(self, tenant_id: str, node_id: str, when: datetime) -> MemoryNode | None: ...

    @abstractmethod
    def list_nodes(self, tenant_id: str, status: NodeStatus | None = None) -> list[MemoryNode]: ...

    @abstractmethod
    def update_decay_score(self, tenant_id: str, node_id: str, decay_score: float) -> None: ...

    @abstractmethod
    def mark_for_deletion(self, tenant_id: str, node_id: str, when: datetime) -> None: ...

    @abstractmethod
    def reactivate(self, tenant_id: str, node_id: str) -> None: ...

    @abstractmethod
    def hard_delete(self, tenant_id: str, node_id: str) -> bool: ...

    @abstractmethod
    def multi_hop(self, tenant_id: str, start_ids: list[str], hops: int) -> list[HopResult]: ...

    def close(self) -> None:  # pragma: no cover - trivial default
        return None


class InMemoryGraphStore(GraphStore):
    """Pure-python fake adapter implementing the exact same interface as the
    real Neo4j-backed store. Used by unit tests and as the default demo
    backend when ``USE_FAKE_STORES=true`` (no docker required).
    """

    def __init__(self) -> None:
        self._nodes: dict[tuple[str, str], MemoryNode] = {}
        # adjacency[(tenant, src)] = list[(relation, target)]
        self._adjacency: dict[tuple[str, str], list[tuple[str, str]]] = {}
        self._lock = threading.RLock()

    def upsert_node(self, node: MemoryNode) -> None:
        with self._lock:
            self._nodes[(node.tenant_id, node.id)] = node.model_copy(deep=True)

    def get_node(self, tenant_id: str, node_id: str) -> MemoryNode | None:
        with self._lock:
            node = self._nodes.get((tenant_id, node_id))
            return node.model_copy(deep=True) if node else None

    def upsert_edge(self, edge: GraphEdge) -> None:
        with self._lock:
            key = (edge.tenant_id, edge.source_id)
            self._adjacency.setdefault(key, [])
            entry = (edge.relation, edge.target_id)
            if entry not in self._adjacency[key]:
                self._adjacency[key].append(entry)

    def record_access(self, tenant_id: str, node_id: str, when: datetime) -> MemoryNode | None:
        with self._lock:
            node = self._nodes.get((tenant_id, node_id))
            if node is None:
                return None
            node.access_count += 1
            node.last_accessed_at = when
            return node.model_copy(deep=True)

    def list_nodes(self, tenant_id: str, status: NodeStatus | None = None) -> list[MemoryNode]:
        with self._lock:
            result = [
                n.model_copy(deep=True)
                for (t, _nid), n in self._nodes.items()
                if t == tenant_id and (status is None or n.status == status)
            ]
        return result

    def update_decay_score(self, tenant_id: str, node_id: str, decay_score: float) -> None:
        with self._lock:
            node = self._nodes.get((tenant_id, node_id))
            if node is not None:
                node.decay_score = decay_score

    def mark_for_deletion(self, tenant_id: str, node_id: str, when: datetime) -> None:
        with self._lock:
            node = self._nodes.get((tenant_id, node_id))
            if node is not None:
                node.status = NodeStatus.MARKED_FOR_DELETION
                node.marked_at = when

    def reactivate(self, tenant_id: str, node_id: str) -> None:
        with self._lock:
            node = self._nodes.get((tenant_id, node_id))
            if node is not None:
                node.status = NodeStatus.ACTIVE
                node.marked_at = None

    def hard_delete(self, tenant_id: str, node_id: str) -> bool:
        with self._lock:
            existed = self._nodes.pop((tenant_id, node_id), None) is not None
            self._adjacency.pop((tenant_id, node_id), None)
            # remove dangling edges pointing at the deleted node
            for key, edges in list(self._adjacency.items()):
                if key[0] != tenant_id:
                    continue
                self._adjacency[key] = [e for e in edges if e[1] != node_id]
            return existed

    def multi_hop(self, tenant_id: str, start_ids: list[str], hops: int) -> list[HopResult]:
        with self._lock:
            visited: dict[str, HopResult] = {}
            frontier: deque[tuple[str, int, list[str]]] = deque(
                (sid, 0, []) for sid in start_ids
            )
            while frontier:
                current_id, depth, path = frontier.popleft()
                if depth >= hops:
                    continue
                for relation, target_id in self._adjacency.get((tenant_id, current_id), []):
                    if target_id in visited:
                        continue
                    node = self._nodes.get((tenant_id, target_id))
                    if node is None or node.status == NodeStatus.DELETED:
                        continue
                    new_path = path + [relation]
                    visited[target_id] = HopResult(
                        node=node.model_copy(deep=True), hops=depth + 1, relation_path=new_path
                    )
                    frontier.append((target_id, depth + 1, new_path))
            return list(visited.values())


class Neo4jGraphStore(GraphStore):
    """Real Neo4j-backed implementation.

    Tenants are partitioned by a ``tenant_id`` node property (+ index) by
    default, since the docker-compose demo runs Neo4j Community Edition
    (single database). When ``multi_db_enabled`` is true *and* the server
    supports it (Enterprise/AuraDB), each tenant is routed to its own
    database named via :func:`app.tenancy.neo4j_database_name`.
    """

    def __init__(self, uri: str, user: str, password: str, multi_db_enabled: bool = False) -> None:
        from neo4j import GraphDatabase  # imported lazily so fake-store users
        # don't need the `neo4j` package installed at all.

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._multi_db_enabled = multi_db_enabled
        self._ensure_constraints()

    def _database_for(self, tenant_id: str) -> str | None:
        return neo4j_database_name(tenant_id) if self._multi_db_enabled else None

    def _ensure_constraints(self) -> None:
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT memory_node_id IF NOT EXISTS "
                "FOR (n:MemoryNode) REQUIRE (n.tenant_id, n.id) IS NODE KEY"
            )

    def close(self) -> None:
        self._driver.close()

    def upsert_node(self, node: MemoryNode) -> None:
        with self._driver.session(database=self._database_for(node.tenant_id)) as session:
            session.run(
                """
                MERGE (n:MemoryNode {tenant_id: $tenant_id, id: $id})
                SET n.kind = $kind, n.text = $text, n.base_importance = $base_importance,
                    n.last_accessed_at = $last_accessed_at, n.created_at = $created_at,
                    n.access_count = $access_count, n.pinned = $pinned,
                    n.decay_score = $decay_score, n.status = $status, n.marked_at = $marked_at
                """,
                tenant_id=node.tenant_id,
                id=node.id,
                kind=node.kind.value,
                text=node.text,
                base_importance=node.base_importance,
                last_accessed_at=node.last_accessed_at.isoformat(),
                created_at=node.created_at.isoformat(),
                access_count=node.access_count,
                pinned=node.pinned,
                decay_score=node.decay_score,
                status=node.status.value,
                marked_at=node.marked_at.isoformat() if node.marked_at else None,
            )

    def upsert_edge(self, edge: GraphEdge) -> None:
        with self._driver.session(database=self._database_for(edge.tenant_id)) as session:
            session.run(
                """
                MATCH (a:MemoryNode {tenant_id: $tenant_id, id: $source_id})
                MATCH (b:MemoryNode {tenant_id: $tenant_id, id: $target_id})
                MERGE (a)-[r:RELATES {type: $relation}]->(b)
                SET r.weight = $weight
                """,
                tenant_id=edge.tenant_id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                relation=edge.relation,
                weight=edge.weight,
            )

    def get_node(self, tenant_id: str, node_id: str) -> MemoryNode | None:
        with self._driver.session(database=self._database_for(tenant_id)) as session:
            record = session.run(
                "MATCH (n:MemoryNode {tenant_id: $tenant_id, id: $id}) RETURN n",
                tenant_id=tenant_id,
                id=node_id,
            ).single()
            return self._record_to_node(record["n"]) if record else None

    def record_access(self, tenant_id: str, node_id: str, when: datetime) -> MemoryNode | None:
        with self._driver.session(database=self._database_for(tenant_id)) as session:
            record = session.run(
                """
                MATCH (n:MemoryNode {tenant_id: $tenant_id, id: $id})
                SET n.access_count = n.access_count + 1, n.last_accessed_at = $when
                RETURN n
                """,
                tenant_id=tenant_id,
                id=node_id,
                when=when.isoformat(),
            ).single()
            return self._record_to_node(record["n"]) if record else None

    def list_nodes(self, tenant_id: str, status: NodeStatus | None = None) -> list[MemoryNode]:
        query = "MATCH (n:MemoryNode {tenant_id: $tenant_id})"
        params: dict = {"tenant_id": tenant_id}
        if status is not None:
            query += " WHERE n.status = $status"
            params["status"] = status.value
        query += " RETURN n"
        with self._driver.session(database=self._database_for(tenant_id)) as session:
            result = session.run(query, **params)
            return [self._record_to_node(r["n"]) for r in result]

    def update_decay_score(self, tenant_id: str, node_id: str, decay_score: float) -> None:
        with self._driver.session(database=self._database_for(tenant_id)) as session:
            session.run(
                "MATCH (n:MemoryNode {tenant_id: $tenant_id, id: $id}) SET n.decay_score = $decay_score",
                tenant_id=tenant_id,
                id=node_id,
                decay_score=decay_score,
            )

    def mark_for_deletion(self, tenant_id: str, node_id: str, when: datetime) -> None:
        with self._driver.session(database=self._database_for(tenant_id)) as session:
            session.run(
                """
                MATCH (n:MemoryNode {tenant_id: $tenant_id, id: $id})
                SET n.status = $status, n.marked_at = $when
                """,
                tenant_id=tenant_id,
                id=node_id,
                status=NodeStatus.MARKED_FOR_DELETION.value,
                when=when.isoformat(),
            )

    def reactivate(self, tenant_id: str, node_id: str) -> None:
        with self._driver.session(database=self._database_for(tenant_id)) as session:
            session.run(
                """
                MATCH (n:MemoryNode {tenant_id: $tenant_id, id: $id})
                SET n.status = $status, n.marked_at = null
                """,
                tenant_id=tenant_id,
                id=node_id,
                status=NodeStatus.ACTIVE.value,
            )

    def hard_delete(self, tenant_id: str, node_id: str) -> bool:
        with self._driver.session(database=self._database_for(tenant_id)) as session:
            result = session.run(
                "MATCH (n:MemoryNode {tenant_id: $tenant_id, id: $id}) DETACH DELETE n RETURN count(n) AS c",
                tenant_id=tenant_id,
                id=node_id,
            ).single()
            return bool(result and result["c"] > 0)

    def multi_hop(self, tenant_id: str, start_ids: list[str], hops: int) -> list[HopResult]:
        if hops <= 0 or not start_ids:
            return []
        with self._driver.session(database=self._database_for(tenant_id)) as session:
            # NOTE: hops is interpolated directly (not bound as a $param)
            # because Cypher variable-length path bounds must be literals.
            query = f"""
                MATCH (start:MemoryNode {{tenant_id: $tenant_id}})
                WHERE start.id IN $start_ids
                MATCH path = (start)-[r:RELATES*1..{int(hops)}]->(end:MemoryNode {{tenant_id: $tenant_id}})
                WHERE end.status <> $deleted
                RETURN end, length(path) AS hops, [rel IN r | rel.type] AS relation_path
                """
            result = session.run(
                query,
                tenant_id=tenant_id,
                start_ids=start_ids,
                deleted=NodeStatus.DELETED.value,
            )
            out: dict[str, HopResult] = {}
            for record in result:
                node = self._record_to_node(record["end"])
                if node.id not in out or record["hops"] < out[node.id].hops:
                    out[node.id] = HopResult(
                        node=node, hops=record["hops"], relation_path=list(record["relation_path"])
                    )
            return list(out.values())

    @staticmethod
    def _record_to_node(raw) -> MemoryNode:
        data = dict(raw)
        data["kind"] = data["kind"]
        return MemoryNode(**data)
