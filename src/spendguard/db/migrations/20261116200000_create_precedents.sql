create table precedents (
    id uuid primary key default gen_random_uuid(),
    subject text not null,
    code text not null,
    clause_reference text not null,
    clause_hash text not null,
    reason text not null,
    approver text not null,
    submission_id uuid not null references submissions (id),
    recorded_at timestamptz not null default clock_timestamp(),
    unique (submission_id, code, clause_reference)
);

create index precedents_subject_code_recorded_at on precedents (subject, code, recorded_at desc);
