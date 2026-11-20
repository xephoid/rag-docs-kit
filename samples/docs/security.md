# Security Overview

Tidepool is designed so that your files never leave infrastructure you control.

## Encryption

All traffic between clients and the server uses TLS 1.3. Files are encrypted at
rest with AES-256-GCM using a per-workspace key. Workspace keys are wrapped by a
server master key stored in `/etc/tidepool/master.key`, which should be backed up
separately from the data directory.

## Authentication

Users sign in with email and password, and admins can require two-factor
authentication with TOTP apps for every member of a workspace. Business plan
customers can enable single sign-on through SAML 2.0 or OpenID Connect.

### Session Lifetime

Desktop client sessions stay signed in for 90 days. Web console sessions expire
after 12 hours of inactivity. Admins can revoke all sessions for a user from the
**Members** page, which signs them out on every device within one minute.

## Audit Logs

On the Business plan, audit logs record sign-ins, sharing changes, and admin
actions. Logs are retained for one year and can be exported as CSV.
