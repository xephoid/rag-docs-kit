# Configuration Reference

The Tidepool server reads its settings from `/etc/tidepool/tidepool.toml`.
Restart the service after editing the file: `sudo systemctl restart tidepool`.

## Network Settings

The `listen_port` key controls the port of the web console and sync API. The
default is 8443. Set `public_url` to the address clients use to reach the server
when it sits behind a reverse proxy.

## Storage Settings

The `data_dir` key sets where synced files are stored; the default is
`/var/lib/tidepool`. File versions are kept for 30 days by default. Change the
retention window with the `version_retention_days` key. Setting it to 0 disables
version history entirely.

### Storage Quotas

Each workspace has a quota measured in gigabytes, set with `default_quota_gb`
(default 100). When a workspace reaches 90 percent of its quota, admins receive
a warning email. At 100 percent, new uploads are rejected until space is freed.

## Sync Behavior

Clients poll the server every 15 seconds by default, controlled by
`poll_interval_seconds`. Files larger than `max_file_size_mb` (default 4096) are
skipped and reported in the client activity log. Selective sync lets each user
exclude folders from their device without removing them from the workspace.
