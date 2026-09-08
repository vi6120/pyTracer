# navik-registry

Layer 5 of Navik: turns test collections into shareable, forkable artifacts —
the role Postman's collections played for API testing. Collections are
versioned (content-addressed history), forkable with lineage, mergeable with
conflict detection, access-controlled (private / team / public), and
discoverable by framework and use case.

```python
from navik_registry import Registry, Visibility, CollectionArtifact

reg = Registry()  # in-memory; pass a FilesystemStore for on-disk JSON
col = reg.create_collection(
    "alice", CollectionArtifact(name="crewai-rag-tests", framework="crewai"),
    visibility=Visibility.PUBLIC,
)
fork = reg.fork("bob", col.id)                 # independent copy with lineage
reg.commit("bob", fork.id, updated_artifact, "tweak assertions")
pr = reg.open_pull_request("bob", fork.id, col.id, "improve RAG coverage")
reg.merge_pull_request("alice", pr.id)         # 3-way merge, raises on conflict
reg.discover(viewer="carol", framework="crewai")
```

Storage is behind a `Store` interface (`InMemoryStore`, `FilesystemStore`); a
production deployment can back it with a git repo per collection and serve the
same API to a web UI.
