def require_optimizer_updates(telemetry):
    updates = sum(not row['skipped'] for row in telemetry)
    if not updates:
        raise RuntimeError('Audit has zero actual optimizer updates; synchronized AMP skips are insufficient')
    return updates
