alter table historical_claims
    add column quantity integer not null default 1 check (quantity >= 1);

alter table historical_claims
    alter column quantity drop default;
