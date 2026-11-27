# 0001 Service Runtime

Date: 2026-08-17

## Status

Accepted

## Context

The application must call `service.run`.

## Decision

Use `service.run` from the entry point.

## Consequences

Positive:

- Keeps `Service` construction explicit.

Tradeoffs:

- The service API is public.
