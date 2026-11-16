create table policy_chunks (
    clause_reference text not null,
    chunk_index integer not null check (chunk_index >= 0),
    heading_path text[] not null,
    text text not null,
    clause_hash text not null,
    policy_version text not null,
    embedding vector(1024) not null,
    primary key (clause_reference, chunk_index)
);

create index policy_chunks_embedding on policy_chunks using hnsw (embedding vector_cosine_ops);
