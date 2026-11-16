delete from policy_chunks;

alter table policy_chunks add column embedding_model_id text not null;
