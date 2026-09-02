#!/usr/bin/env bash
# One-shot Samba share for the RTX 5090 box, so a Mac can mount the home
# directory the way it mounted the GB10 (smb://spark-5208.local/sharaths).
#
# Run once, with sudo, on the 5090:
#   sudo bash scripts/setup_smb_share.sh
# Then on the Mac: Finder > Go > Connect to Server > smb://ss-Fusion-75.local/ss
# (user: ss, password: the one smbpasswd asks you for below).
#
# What it does: installs samba, writes a macOS-friendly smb.conf exposing
# /home/ss as share "ss" to user ss only (no guest access), sets the SMB
# password interactively, enables smbd, and opens 445/tcp in ufw if ufw is on.
set -euo pipefail

SHARE_USER="${SHARE_USER:-ss}"
SHARE_PATH="${SHARE_PATH:-/home/$SHARE_USER}"
SHARE_NAME="${SHARE_NAME:-$SHARE_USER}"

[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
id "$SHARE_USER" >/dev/null 2>&1 || { echo "no such user: $SHARE_USER"; exit 1; }

echo "== installing samba =="
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq samba >/dev/null

echo "== writing /etc/samba/smb.conf (previous copy kept as smb.conf.bak) =="
[ -f /etc/samba/smb.conf ] && cp /etc/samba/smb.conf /etc/samba/smb.conf.bak
cat > /etc/samba/smb.conf <<EOF
[global]
   workgroup = WORKGROUP
   server string = $(hostname) (RTX 5090)
   server role = standalone server
   security = user
   map to guest = never
   server min protocol = SMB2
   # macOS Finder compatibility (resource forks, metadata, Time Machine-safe)
   vfs objects = fruit streams_xattr
   fruit:metadata = stream
   fruit:model = MacSamba
   fruit:veto_appledouble = no
   fruit:nfs_aces = no
   fruit:wipe_intentionally_left_blank_rfork = yes
   fruit:delete_empty_adfiles = yes
   ea support = yes
   # Only the LAN
   interfaces = lo $(ip -o -4 route show to default | awk '{print $5}' | head -1)
   bind interfaces only = yes
   log file = /var/log/samba/log.%m
   max log size = 1000
   logging = file

[$SHARE_NAME]
   path = $SHARE_PATH
   comment = $SHARE_USER home on the 5090
   browseable = yes
   read only = no
   valid users = $SHARE_USER
   force user = $SHARE_USER
   create mask = 0644
   directory mask = 0755
EOF
testparm -s >/dev/null

echo "== SMB password for user $SHARE_USER (asked by smbpasswd, stored hashed by Samba) =="
smbpasswd -a "$SHARE_USER"

echo "== enabling and starting smbd =="
systemctl enable --now smbd
systemctl restart smbd

if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
  echo "== opening Samba in ufw (LAN only) =="
  ufw allow from 192.168.0.0/24 to any port 445 proto tcp
fi

echo
echo "Done. On the Mac: Finder > Go > Connect to Server >"
echo "   smb://$(hostname).local/$SHARE_NAME      (or smb://$(hostname -I | cut -d' ' -f1)/$SHARE_NAME)"
echo "   user: $SHARE_USER   password: the one you just set"
systemctl --no-pager --lines=0 status smbd | head -3
