#!/bin/sh
# Switch the robot between the SPRING-BOOT hop setup and NORMAL walking.
# Runs on the LAPTOP.
#
#   sh scripts/robot/mode.sh status          what is the robot running now?
#   sh scripts/robot/mode.sh hop             spring-boot hop setup
#   sh scripts/robot/mode.sh walk            normal operation
#
# Optional second argument is user@host (default microduck@192.168.1.25).
#
# WHY A SINGLE ENTRY POINT. There were two scripts, deploy_hop.sh and
# restore_walk.sh, and they were not inverses: hop set `sitstand = "none"` and
# walk never cleared it, so "restore walking" silently left sit/stand disabled
# from 2026-09-08 onward. One command with two branches makes that class of
# asymmetry visible, and `status` lets you check rather than assume.
#
# The modes differ in more than the policy file:
#
#   key             hop            walk       why
#   gain            400            200        hop needs the stiffer firmware P
#   action_scale    1.0            0.9
#   legs_lowpass    0.0            0.7        training is UNFILTERED; 0.7 at
#                                             50 Hz cuts near 1.4 Hz and guts a
#                                             ~5 Hz gait. 0.0 is the matched value.
#   limp_fall       false          true       a hopping robot should not go limp
#   sitstand        "none"         (unset)    the sitstand policy was trained
#                                             without boots and rises against a
#                                             floor the boots change
#   padd            stopped        started    two writers to one intent fight;
#                                             the hop driver owns the pad
set -e
MODE="${1:-status}"
ROBOT="${2:-microduck@192.168.1.25}"
HERE=$(cd "$(dirname "$0")/../.." && pwd)

ssh -o ConnectTimeout=8 "$ROBOT" true || { echo "robot unreachable: $ROBOT"; exit 1; }

case "$MODE" in
  status)
    echo "== $ROBOT"
    ssh "$ROBOT" '
      echo "robotd: $(systemctl is-active robotd)   padd: $(systemctl is-active padd)"
      echo
      grep -E "^(walk|stand|gain|action_scale|legs_lowpass|head_lowpass|limp_fall|cmd_alpha|sitstand|ground_pick) " \
        /etc/robot/robotd.toml 2>/dev/null
      echo
      w=$(grep -E "^walk " /etc/robot/robotd.toml | sed "s/.*policies\/current\///;s/\".*//")
      case "$w" in
        alpha_walking.onnx) echo "MODE: walk (normal operation)" ;;
        hop*|Hop*)          echo "MODE: hop (spring boot)" ;;
        *)                  echo "MODE: unrecognised -- walk slot holds $w" ;;
      esac'
    ;;
  hop)
    sh "$HERE/scripts/robot/deploy_hop.sh" "$ROBOT"
    echo
    echo "hop mode. padd is stopped; start the driver with ~/start_hop_driver.sh"
    ;;
  walk)
    sh "$HERE/scripts/robot/restore_walk.sh" "$ROBOT"
    ;;
  *)
    echo "usage: mode.sh {status|hop|walk} [user@host]" >&2
    exit 2
    ;;
esac
