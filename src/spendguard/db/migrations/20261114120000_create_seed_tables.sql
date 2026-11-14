create table employees (
    reference text primary key,
    grade text not null,
    cost_centre text not null,
    manager text references employees (reference) deferrable initially deferred
);

create table cost_centres (
    reference text primary key,
    owner text references employees (reference) deferrable initially deferred,
    budget_gbp numeric(14, 2) not null check (budget_gbp > 0),
    spend_to_date_gbp numeric(14, 2) not null check (spend_to_date_gbp >= 0)
);

alter table employees
    add foreign key (cost_centre) references cost_centres (reference) deferrable initially deferred;

create table approvers (
    employee text primary key references employees (reference) deferrable initially deferred,
    approval_limit_gbp numeric(14, 2) not null check (approval_limit_gbp > 0)
);

create table suppliers (
    reference text primary key,
    approved boolean not null
);

create table purchase_orders (
    reference text primary key,
    raised_by text not null references employees (reference) deferrable initially deferred,
    supplier text not null references suppliers (reference) deferrable initially deferred,
    cost_centre text not null references cost_centres (reference) deferrable initially deferred,
    amount numeric(14, 2) not null check (amount > 0),
    currency char(3) not null
);

create table historical_claims (
    reference text primary key,
    claimant text not null references employees (reference) deferrable initially deferred,
    spent_on date not null,
    category text not null,
    merchant text not null,
    amount numeric(14, 2) not null check (amount > 0),
    currency char(3) not null
);

create table historical_invoices (
    reference text primary key,
    supplier text not null references suppliers (reference) deferrable initially deferred,
    supplier_invoice_number text not null,
    purchase_order text references purchase_orders (reference) deferrable initially deferred,
    cost_centre text references cost_centres (reference) deferrable initially deferred,
    issued_on date not null,
    amount numeric(14, 2) not null check (amount > 0),
    currency char(3) not null,
    check (purchase_order is null or cost_centre is null)
);

create table fx_rates (
    currency char(3) not null,
    rate_date date not null,
    gbp_per_unit numeric(18, 8) not null check (gbp_per_unit > 0),
    primary key (currency, rate_date)
);

create table public_holidays (
    holiday_on date primary key,
    description text not null
);
