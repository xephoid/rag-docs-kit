# Troubleshooting

This page covers the most common problems reported by Tidepool admins.

## Client Cannot Connect

First confirm the server is running with `systemctl status tidepool`. If the
client shows "certificate not trusted", install a certificate from a public
certificate authority or distribute your internal CA to client machines. Make
sure the firewall allows inbound TCP traffic on the configured listen port.

## Files Are Not Syncing

Open the client activity log from the tray icon. Files marked "skipped" usually
exceed the maximum file size or match an ignore rule. Ignore rules live in a
`.tidepoolignore` file at the root of a synced folder and use gitignore syntax.

### Sync Conflicts

When two people edit the same file before syncing, Tidepool keeps both copies.
The second copy is renamed with the suffix `(conflict <username> <date>)` so no
work is lost. Resolve the conflict by merging the changes and deleting the
extra copy.

## Server Logs

Server logs are written to the systemd journal. View them with
`journalctl -u tidepool`. To collect a support bundle, run
`tidepool-admin support-bundle`, which writes a redacted archive to `/tmp`.
