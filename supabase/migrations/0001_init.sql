-- Kinetic AI app schema — user tiers, job history, runtime config, gateway setup.
-- RLS on every table; public endpoints require Bearer auth; job history is per-user.

create extension if not exists "uuid-ossp";

-- ─── User Tiers ──────────────────────────────────────────────────────────────
-- Bootstrap pattern: admin users get full access; default is 'user'.
-- Seed: sharath.sathish@gmail.com is admin.

create table if not exists public.user_tiers (
  user_id uuid primary key references auth.users(id) on delete cascade,
  tier text not null default 'user' check (tier in ('user', 'admin')),
  created_at timestamptz default now()
);
alter table public.user_tiers enable row level security;

create policy "users_view_own_tier" on public.user_tiers
  for select using (auth.uid() = user_id);

-- Admin bootstrap: grant the admin tier automatically when the admin email
-- signs up. Guests get rows only via the admin RPC below (access = has a row).
create or replace function public.handle_new_user()
returns trigger as $$
begin
  if new.email = 'sharath.sathish@gmail.com' then
    insert into public.user_tiers (user_id, tier)
    values (new.id, 'admin')
    on conflict (user_id) do update set tier = 'admin';
  end if;
  return new;
end;
$$ language plpgsql security definer;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- Admin-only RPC to enable a guest account (same access as admin per operator).
create or replace function public.set_user_tier(target_email text, new_tier text)
returns json as $$
declare
  is_admin boolean;
  target_id uuid;
begin
  select (tier = 'admin') into is_admin
  from public.user_tiers where user_id = auth.uid();
  if not coalesce(is_admin, false) then
    raise exception 'Admin access required';
  end if;
  if new_tier not in ('user', 'admin') then
    raise exception 'Invalid tier';
  end if;
  select id into target_id from auth.users where email = target_email;
  if target_id is null then
    raise exception 'No user with that email (they must sign up first)';
  end if;
  insert into public.user_tiers (user_id, tier)
  values (target_id, new_tier)
  on conflict (user_id) do update set tier = new_tier;
  return json_build_object('success', true, 'email', target_email, 'tier', new_tier);
end;
$$ language plpgsql security definer;

-- ─── Job History ─────────────────────────────────────────────────────────────
-- Training Studio jobs: user submits via API, tracked here per-user.
-- RLS: each user sees only their own jobs.

create table if not exists public.job_history (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid not null references auth.users(id) on delete cascade,
  job_spec jsonb not null,
  status text not null check (status in ('queued', 'running', 'completed', 'failed')),
  result jsonb,
  error text,
  created_at timestamptz default now(),
  completed_at timestamptz
);
alter table public.job_history enable row level security;

create policy "users_view_own_jobs" on public.job_history
  for select using (auth.uid() = user_id);

create policy "users_insert_own_jobs" on public.job_history
  for insert with check (auth.uid() = user_id);

create policy "users_update_own_jobs" on public.job_history
  for update using (auth.uid() = user_id);

create index if not exists job_history_user_id_idx on public.job_history(user_id);
create index if not exists job_history_created_at_idx on public.job_history(created_at);
create index if not exists job_history_status_idx on public.job_history(status);

-- ─── Runtime Config ──────────────────────────────────────────────────────────
-- Gateway URL, secret, and other runtime settings. Public-read for gateway discovery;
-- admin-write only (via RPC).

create table if not exists public.runtime_config (
  key text primary key,
  value text,
  updated_by uuid references auth.users(id),
  updated_at timestamptz default now()
);
alter table public.runtime_config enable row level security;

-- Allowlist: only explicitly-named non-secret keys are publicly readable.
-- Everything else (including gateway_secret and any future key) is admin-only.
create policy "public_read_config" on public.runtime_config
  for select using (key in ('gateway_url', 'brand_name', 'announcement'));

create policy "admin_read_secrets" on public.runtime_config
  for select using (
    exists (select 1 from public.user_tiers
            where user_tiers.user_id = auth.uid() and user_tiers.tier = 'admin'));

-- One action per policy (Postgres requirement); admin gate on each write path.
create policy "admin_insert_config" on public.runtime_config
  for insert with check (
    exists (select 1 from public.user_tiers
            where user_tiers.user_id = auth.uid() and user_tiers.tier = 'admin'));

create policy "admin_update_config" on public.runtime_config
  for update using (
    exists (select 1 from public.user_tiers
            where user_tiers.user_id = auth.uid() and user_tiers.tier = 'admin'));

create policy "admin_delete_config" on public.runtime_config
  for delete using (
    exists (select 1 from public.user_tiers
            where user_tiers.user_id = auth.uid() and user_tiers.tier = 'admin'));

-- RPC to set config (admin only; security definer bypasses RLS after the check)
create or replace function public.set_runtime_config(key_name text, key_value text)
returns json as $$
declare
  is_admin boolean;
begin
  select (tier = 'admin') into is_admin
  from public.user_tiers where user_id = auth.uid();

  if not coalesce(is_admin, false) then
    raise exception 'Admin access required';
  end if;

  insert into public.runtime_config (key, value, updated_by)
  values (key_name, key_value, auth.uid())
  on conflict (key) do update
  set value = key_value, updated_by = auth.uid(), updated_at = now();

  return json_build_object('success', true, 'key', key_name);
end;
$$ language plpgsql security definer;

-- Bootstrap non-secret config keys only. NEVER seed secrets in migrations:
-- set gateway_secret out-of-band via select set_runtime_config('gateway_secret', '<value>');
insert into public.runtime_config (key, value)
values
  ('gateway_url', 'https://kinetic.kinetic-ai.workers.dev')
on conflict (key) do nothing;
