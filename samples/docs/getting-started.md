# Getting Started with Tidepool

Tidepool is a self-hosted file sync service for small teams. A single Tidepool
server keeps shared folders in sync across laptops, desktops, and build machines.

## System Requirements

The Tidepool server runs on Linux (x86_64 or arm64) with at least 1 GB of RAM and
2 GB of free disk space beyond the size of your synced data. The desktop client
supports Windows 10 or later, macOS 13 or later, and Ubuntu 22.04 or later.

## Installation

Download the server package from your account portal and install it with your
package manager, for example `sudo apt install ./tidepool-server.deb`. The installer
creates a `tidepool` system user and a systemd unit named `tidepool.service`.

Start the server with `sudo systemctl enable --now tidepool`. The web console
listens on port 8443 by default.

## Creating Your First Workspace

Open the web console and sign in with the admin password printed during
installation. Choose **New Workspace**, give it a name, and invite teammates by
email. Each workspace has its own folders, members, and storage quota.

### Inviting Teammates

Invitations expire after 7 days. An admin can resend an expired invitation from
the **Members** page. Invited users must install the desktop client and sign in
with the email address that received the invitation.
