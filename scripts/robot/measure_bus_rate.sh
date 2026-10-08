#!/bin/sh
# Measure the real bus ceiling with the bus actually free. Runs ON the robot.
#
#   sudo sh ~/measure_bus_rate.sh
#
# WHY A WRAPPER, same reason as measure_travel.sh: `systemctl stop robotd` does
# not keep robotd stopped. The unit is Restart=always with RestartSec=2s, and
# updaterd's health gate restarts it besides. Measured: stopped t+0, inactive
# t+1..t+3, activating t+4, active t+5. Any measurement longer than ~3 s then
# has robotd back on the bus contending for every reply.
#
# NO `set -e`, deliberately: `systemctl mask` returns non-zero even on success,
# which fired the cleanup trap before the measurement ran. A wrapper whose job
# is "restore state on every exit path" wants an explicit trap.
restore() {
  echo
  echo "restoring robotd..."
  sudo systemctl unmask robotd >/dev/null 2>&1 || true
  sudo systemctl start robotd >/dev/null 2>&1 || true
  sleep 2
  echo "robotd: $(systemctl is-active robotd)"
}
trap restore EXIT INT TERM

echo "masking robotd so it cannot restart onto the bus..."
sudo systemctl mask robotd >/dev/null 2>&1
sudo systemctl stop robotd >/dev/null 2>&1
sleep 3
echo "robotd: $(systemctl is-active robotd)"

# Not `~/bus_rate.py`: under sudo that expands to /root. Resolve it next to
# this script instead.
DIR=$(dirname "$0")
sudo python3 "$DIR/bus_rate.py" "$@"
