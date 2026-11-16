create table submissions (
    id uuid primary key default gen_random_uuid(),
    idempotency_key text not null unique,
    submission_type text not null check (submission_type in ('expense_claim', 'supplier_invoice')),
    payload jsonb not null,
    payload_hash text not null,
    status text not null check (status in ('received', 'pending_review', 'decided', 'complete')),
    decision jsonb,
    decided_by text,
    created_at timestamptz not null default now(),
    status_changed_at timestamptz not null default now()
);

create index submissions_status_changed_at on submissions (status, status_changed_at);

create table decision_events (
    idempotency_key text primary key,
    payload jsonb not null,
    received_at timestamptz not null default now()
);
