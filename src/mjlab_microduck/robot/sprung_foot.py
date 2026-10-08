"""Sprung-foot robot model — an idealised 1-DoF compliant foot accessory.

Built PROGRAMMATICALLY from the canonical ``robot_walk.xml`` rather than as a
forked XML. The abandoned ``test_spring`` branch forked the XML, and its 50-line
delta became unusable once ``robot_walk.xml`` moved by 310 insertions. Adding two
bodies to the live spec tracks every upstream change to the base model for free.

The mechanism modelled here is deliberately idealised: one prismatic spring per
foot. That is not a shortcut — it is the design target. A rigid 1-DoF
translating mechanism (a prismatic slide, or a Sarrus linkage) maps exactly onto
a MuJoCo ``slide`` joint, so the kinematics carry no sim-to-real gap. A
Kangoo-style leaf flexure would need a discretised multi-body chain or
deformables, and was rejected on that basis. See the Phase 2 spec.
"""

from __future__ import annotations

from typing import Callable

import math

import mujoco
import numpy as np

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg

from mjlab_microduck.robot.microduck_constants import (
    FULL_COLLISION,
    HOME_FRAME,
    actuators,
    get_allcollisions_spec,
)

# Local +y of the ankle bodies maps to world [0, 0.087, 0.996] — almost straight
# up. So a slide along +y means positive q = compression (pad moves toward the
# body). Local +z is nearly HORIZONTAL; using it would slide the foot sideways.
SPRING_AXIS = (0.0, 1.0, 0.0)

# Distance from the ankle body origin down to the existing sole's contact plane,
# measured at the home pose. The pad is placed h_add BELOW that, which is what
# makes the sprung robot taller than the rigid one.
#
# Tuned (not the naive mesh measurement of 0.025) so that the settled
# rigid-vs-sprung trunk-height delta lands on H_ADD: the rigid sole is a mesh
# and the pad is a box, and the two settle to slightly different contact
# penetration depths under gravity, so the naive value overshot the delta by
# ~3-4 mm. See task-1-report.md fix-round-1 notes.
#
# Retuned for the measured H_ADD=0.030 (was 0.0215, tuned for the old
# H_ADD=0.025): the mesh-vs-box penetration mismatch this constant corrects
# for is unaffected by H_ADD, so it re-manifested as the same ~4 mm overshoot
# on the LOCKED (zero-compliance) arm's settled delta and was re-tuned down
# by that amount. See FIX 5's settling measurement in the prototype-update
# report.
#
# RETUNED AGAIN 2026-09-04, +3.53 mm, when the pad footprint was corrected to
# the measured 25 x 40 mm. THIS CONSTANT IS COUPLED TO THE FOOTPRINT and that is
# not obvious: it exists to cancel the contact-penetration difference between
# the mesh sole and the box pad, and penetration depends on contact PRESSURE.
# Shrinking the sole from 40x28 to 25x40 mm cut its area 11% and pushed the
# settled Locked-minus-Standard delta to 26.47 mm against the 30.00 mm H_ADD
# target. Anyone touching `_PAD_HALF_EXTENTS` must re-measure that delta and
# re-tune this; `test_sprung_foot.py` pins it so the drift cannot pass silently.
ANKLE_TO_SOLE = 0.02097

H_ADD = 0.030      # measured on the Sarrus prototype (was an assumed 0.025)
# MEASURED stiffness of the prototype boot, 2026-09-03, on the RobStride gripper
# bench: 3344 N/m (two runs 3.6% apart, r^2 > 0.998). Method: jaw position from
# the encoder against a scale in series with the boot -- motor torque is not in
# the measurement, so it is immune to the rack friction and cogging that spoil
# a torque-derived figure (that route gave 5026 N/m, r^2 0.91; discarded).
#
# The earlier hand figure of 3900 N/m (3 mm ~ 1500 g, 8 mm ~ 3500 g) is 17% HIGH.
# The registered hop arms are k2500 and k3900, which bracket 3344, and the sweep
# gave 23 mm and 27 mm of rise respectively -- so ~25-26 mm is the honest
# expectation for the real spring. Not worth re-running for; do it when the load
# cells land a real damping number.
#
# Damping was NOT measured, only bounded: c <= 12.5 N.s/m, i.e. zeta <= 0.41 vs
# the pad / <= 0.115 vs the whole robot. DAMPING_RATIO = 0.3 below gives
# c = 9.18 N.s/m (zeta_eff 0.12 vs the robot), which sits INSIDE that bound -- so
# the assumption is not contradicted and is, if anything, slightly pessimistic.
# Loss was also rate-independent (+5.5% over a 20x frequency span), so the
# viscous term is small. The boot's own Coulomb friction is negligible: it
# returns fully to free length, where 6.29 N in its load path would leave a
# 1.88 mm stiction dead band.
#
# See rebot-lerobot/bench/RESULTS.md and
# docs/sim2real/spring_boot_identification_spec.md.
# SUPERSEDED 2026-10-02 by the load-cell bench, which measured force directly
# at the jaw instead of inferring it. k = 4290 N/m (closing 4360, opening 4220).
# The old 3344 was 22% LOW: it came through the previous 100 mm printed fingers,
# whose compliance was in series with the boot. The rig is now 14x stiffer than
# the boot, so that correction moves k by only 7%. Quote to +/-5%.
# See rebot-lerobot/bench/RESULTS_loadcell.md.
K_MEASURED = 4290.0

# A DELIBERATELY SOFTER BOOT, to test whether the spring can participate AT ALL.
#
# The measured boot cannot be charged by this robot, and the arithmetic is not
# close. A series spring between foot and ground is loaded only by the ground
# reaction force. To store the 0.34 J a 40 mm hop needs, k=4290 must compress
# 8.2 mm, which takes 38 N per boot -- NINE times body weight. The legs deliver
# ~1x statically and ~2x in their best landing, and the max-effort open-loop
# sweep tops out at 4.79 mm of rise (0.217 m/s takeoff), so the hard landing
# that would charge the spring requires a hop the robot cannot make. The loop
# never closes, and every run since 2026-09 has duly measured the boot
# contributing a few percent: loaded compression 0.81-0.89 mm of 12 mm travel,
# p95 1.7-2.4 mm, bottomed 0.0% of the time, unchanged across the stiffness
# correction, the geometry correction and the reward rewrite.
#
# At 1200 N/m full travel takes 15.3 N, about 3.6x body weight, which a 0.6 m/s
# landing actually produces -- and full travel then stores 86 mJ per boot, worth
# ~20 mm of CoM rise against the 5-7 mm the legs manage alone. This is the first
# stiffness where the spring could do something the legs cannot.
#
# SPRING_PRELOAD is a DISPLACEMENT, so softening fixes the preload problem for
# free: preload force falls from 3.17 N to 0.89 N per boot. At k=4290 three
# quarters of each boot's share of body weight was spent overcoming preload
# before the spring moved at all.
K_SOFT = 1200.0

# BRACKETS FOR THE STIFFNESS SWEEP. K_SOFT was chosen off a design table as the
# softest rate the legs can fully charge, not by search, so it locates a regime
# rather than an optimum. These two bracket it across a 5x range together with
# the measured 4290, so the deliverable is a BAND to shop a real boot against
# instead of a single number no off-the-shelf spring will hit.
#
# Why bracket rather than trust 1200: it bottoms out 2.5% of steps, so it is
# near the soft limit already. If 800 bottoms badly the band runs upward from
# 1200; if 2000 holds up, the part is far easier to source.
K_SWEEP_LOW = 800.0
K_SWEEP_HIGH = 2000.0

# BRACKETING THE OPTIMUM FROM ABOVE, which the sweep so far does not.
#
# Measured, strict zero-contact definition, same reward and mass throughout:
#   rigid 7.4 mm  <  k=1200 16.3 mm  <  k=4290 24.0 mm
# monotonically increasing in stiffness, so the peak is at or above the boot
# that exists and every registered sweep arm (800, 2000) is on the wrong side.
#
# THE THEORY PREDICTS 8000 WILL BE WORSE, and that is why it is worth running.
# At a fixed force ceiling E* = (F^2 - (kp)^2)/2k falls with k, and the preload
# penalty grows: at 8000 N/m the preload alone is 5.92 N per boot, 1.39x each
# boot's share of body weight, so the spring is rigid until the load exceeds
# that. Predicted reachable compression drops 2.63 -> 1.07 mm and stored energy
# 34.1 -> 21.9 mJ. If 8000 still beats 4290, transmission dominates storage and
# the search should continue upward. If it loses, the optimum is bracketed and
# the built boot is at or near it. Either outcome ends the question.
K_SWEEP_STIFF = 8000.0

# DELTA mass of fitting a spring boot, per foot: the 69 g spring boot REPLACES
# the 18 g standard pad foot, so 69 - 18 = 51 g. The common motor-to-boot
# interface (16.5 g) is present in BOTH configurations and cancels.
#
# This is a delta, not the boot's mass, because the pad body is ADDED on top of
# the stock model (737.2 g -> compiles with both pads on top) and the stock model
# already matches the real bare robot, i.e. it already contains the standard
# pads. Using the boot's full 70 g over-counted by 19 g per foot.
PAD_MASS = 0.051   # measured 2026-09-03: 69 g spring boot - 18 g standard pad
TRAVEL = 0.012     # measured (was an assumed 0.015)
# Damping is specified as a RATIO, not an absolute rate, and derived per arm as
# c = 2*zeta*sqrt(k*pad_mass).
#
# Why: an absolute c=0.5 N.s/m ("a good steel spring, low hysteresis") is the
# right figure for the SPRING but leaves the PAD-ON-SPRING subsystem essentially
# undamped — zeta came out at 0.013-0.023, the pad rang at 33-57 Hz against a
# 50 Hz controller, and it retained 65-87% of its amplitude across a 51 ms
# stance, so it never settled between steps. The 30 g pad rang ABOVE the control
# rate entirely. In the first Stage 1 sweep this made sprung speed *improve*
# monotonically with pad mass (lighter pad = faster ringing = worse), the
# opposite of the locked arms' trend — resonance masquerading as a mass effect.
#
# A ratio also keeps resonance CONSTANT across a mass sweep instead of letting it
# confound the axis, and is physically defensible: a larger mechanism carries
# proportionally more joint friction. Reaching zeta=0.3 needs c = 6.5-11 N.s/m
# across the 30-90 g range, i.e. 13-22x the old absolute value.
#
# This is provisional. The real number is measurable on the prototype as
# loading-vs-unloading hysteresis; that measurement should replace this estimate.
# SUPERSEDED 2026-10-02. The load-cell bench measured the loss directly and it
# is COULOMB, not viscous, and far smaller than assumed:
#
#   loss per cycle   7% of input work (retention 0.93-0.96 at 5 +/- 2.8 mm)
#   Coulomb Fc       ~0.65 N at that bias, ~4% of load, growing with force
#   viscous c        not detectable, <= ~4 N.s/m, consistent with ZERO
#
# September's "retention 0.35-0.47" was the RACK, not the boot. The boot is far
# springier than this campaign has been assuming.
#
# The sign of the frequency dependence settles the form: loop area FALLS with
# speed (7.64 mJ at 0.1 Hz, 4.70 mJ at 2 Hz). A viscous term would make it
# RISE. Fitting a viscous model to that gives a negative c, because the model
# has no Stribeck term and books the drop as negative viscosity.
#
# So the loss moves to SPRING_FRICTIONLOSS below and the viscous term drops to a
# token value. zeta = 0.3 (c = 9.18 N.s/m) is EXCLUDED by the measurement: for it
# to be right the 2 Hz loop would have to be ~10.7 mJ, needing a lag 5 ms from
# the measured -4.2 +/- 1 ms. It over-damped the boot by about 3x.
DAMPING_RATIO = 0.03

# Absolute now, not ratio-derived. 0.5 N.s/m is a steel spring's own viscous
# loss, well inside the <= 4 N.s/m bound. The old worry that a low viscous term
# leaves the pad ringing at 33-57 Hz against a 50 Hz controller is answered by
# the Coulomb term instead: 0.65 N on a 51 g pad is a 0.15 mm dead band at
# k = 4290, which is what actually stops the chatter in the real mechanism.
DAMPING = 0.5      # absolute N.s/m; None would derive it from DAMPING_RATIO


def damping_for(stiffness: float, pad_mass: float, ratio: float = DAMPING_RATIO) -> float:
    """Critical-damping-scaled rate: c = 2*zeta*sqrt(k*m) for the pad on its spring."""
    return 2.0 * ratio * math.sqrt(stiffness * pad_mass)

# Intentional spring preload in the Sarrus mechanism, as a DISPLACEMENT.
# Measured: 2.9 N offset at k = 3920 N/m -> 2.9/3920 = 0.74 mm of precompression.
# Parameterised as displacement rather than force because the linkage geometry
# fixes the precompression at assembly: a stiffer spring in the same boot keeps
# the 0.74 mm and produces proportionally MORE preload force. Consequence, which
# is physically faithful rather than a modelling artifact: preload force varies
# across the sweep (1.1 N at k=1500, 4.1 N at k=5500).
# Preload holds the pad firmly at full extension during flight instead of
# letting it float within its travel and chatter against the hard stop.
SPRING_PRELOAD = 0.00074   # m of precompression at rest

# These override the `microduck` childclass joint defaults (frictionloss=0.1,
# armature=0.005 in robot_walk.xml), which the spring joint would otherwise
# inherit silently, since it is added inside that childclass scope.
#
# frictionloss is no longer zero, and that is the main modelling change of
# 2026-10-02: the boot's dissipation is Coulomb friction of ~0.65 N, measured
# directly, not the viscous damper this file assumed for a year. It was zero
# before because the spec idealised the spring and deferred stiction to the
# hardware phase -- which has now happened.
#
# 0.65 N is the figure at the 5 mm / ~21 N bias point. It GROWS with load
# (0.33 N at 5 N, 1.0 N at 20-25 N, i.e. ~4% of force), which MuJoCo's constant
# frictionloss cannot express; 0.65 is right in the middle of the robot's own
# stance loads.
SPRING_FRICTIONLOSS = 0.65
SPRING_ARMATURE = 0.0

# Joint-limit solver settings for the mechanical end-stop.
#
# MuJoCo's default limit ([0.02, 1] with dt=0.002, i.e. a 10*dt time constant) is
# soft enough that a hard landing squishes straight through it: a drop-rig probe
# drove k=1500 to 149% of its 12 mm travel from a 100 mm drop. A stop that
# stores-and-returns instead of dissipating systematically overstates rebound,
# which matters because bottoming is common in hop and jump regimes.
#
# 0.004 is the solver's stability floor (2*timestep). Tuned empirically, not
# guessed: it cuts worst-case penetration from 149% to 109% of travel, and the
# stiff arms stay well inside range. MuJoCo cannot make this stop truly rigid at
# a 2 ms timestep, so a residual ~9% overshoot remains under the worst impacts.
# Read a high `spring_bottomed_fraction` as "this arm bottoms out" rather than
# trusting its rebound magnitude to the millimetre.
SPRING_SOLREF_LIMIT = (0.004, 1.0)          # (timeconst, dampratio)
SPRING_SOLIMP_LIMIT = (0.9, 0.99, 0.001, 0.5, 2.0)   # dmax 0.95 -> 0.99

SPRING_JOINTS = ("passive_left_foot_spring", "passive_right_foot_spring")

# The `<default class="collision">` block in robot_walk.xml (group=3). The pad's
# contact geom MUST inherit it: `foot_height_scan` rays are restricted to
# `include_geom_groups=(0,)` (terrain only), so a group-0 pad geom would be hit
# by the opposite foot's height ray and reported as ground, corrupting
# `foot_clearance` and `foot_swing_height`.
_COLLISION_CLASS = "collision"

# Contact pad half-extents (m), as (fore-aft, thickness, lateral).
#
# MEASURED on the Sarrus prototype, 2026-09-04: the contact sole is 25 mm
# fore-aft x 40 mm lateral. The linkage eats most of the original foot's
# footprint, so the boot's sole is much SMALLER than the mesh sole it replaces.
#
# THESE WERE TRANSPOSED UNTIL 2026-09-04, and it flattered every landing in the
# campaign. The old (0.020, 0.004, 0.014) gave 40 mm fore-aft x 28 mm lateral --
# 60% MORE fore-aft base than the robot has, and 30% less lateral. Fore-aft is
# the axis that matters: laterally a biped is braced by having two feet spaced
# apart, but nothing resists a fore-aft topple except sole length and the ankle,
# and the CoM sits ~150 mm up. So the simulated robot was landing on a pitch
# footprint it does not own, and `fell_over = 0.875` was measured on the
# forgiving geometry -- the real figure is worse.
#
# Axis mapping, verified against the compiled model at the home pose rather than
# assumed (local y is world-up here, so the middle number is half the thickness):
#
#   local x -> world [-1.000, 0.000, 0.000]  = fore-aft
#   local y -> world [ 0.000, 0.087, 0.996]  = vertical
#   local z -> world [ 0.000, 0.996,-0.087]  = lateral
_PAD_HALF_EXTENTS = (0.0125, 0.004, 0.020)

# WHERE THE BOOT HANGS. Measured 2026-10-06 from the original sole mesh's
# bounding box in the ankle frame (7896 verts, left; mirrored right):
# centre (-0.0070, -0.0186, -0.0147), extent 54.0 x 12.9 x 41.1 mm.
#
# The boot used to hang from the ANKLE JOINT ORIGIN, which is not where the
# foot is. That put the pad 15.7 mm outboard and 6.9 mm aft of the sole it
# replaces, and the simulated stance 129.0 mm wide against the real 97.5 -- 32%
# too wide, on the very axis the robot is least stable. Every sprung policy in
# this campaign trained on that. Steve confirmed the real boot sole is centred
# under the foot sole.
#
# Fore-aft is signed per side because the two ankle frames are mirrored: local
# +x maps to world -x on the left and +x on the right, so a single sign here
# would move the feet in opposite directions. Lateral is the same for both.
#
# The boot's own sole is FLAT and parallel to the foot, which the box already
# models correctly. The 12.9 mm of thickness in the bbox above is the ORIGINAL
# sole's curvature -- a property of the part the boot replaces, not of the boot.
SOLE_OFFSET_FOREAFT = 0.0070    # magnitude; negative on the left, positive on the right
SOLE_OFFSET_LATERAL = -0.0147

# Fore-aft sole length of the CURRENT prototype, and of the next one. The tip
# angle at the boot's CoM height is atan(half_length / 150.9 mm): 4.4 deg at
# 25 mm, which the robot cannot hold passively, 9.1 deg at 50 mm. Kept as
# named lengths because the two boots coexist -- one on the robot, one on the
# printer -- and a policy is trained for one of them.
SOLE_LENGTH_V1 = 0.025
SOLE_LENGTH_V2 = 0.055      # MEASURED on the printed boot 2026-10-06 (was 0.050)

# Lateral sole WIDTH, which until 2026-10-06 was a single hard-coded 40 mm for
# both versions. Steve measured the V2 boot at 55 x 30 mm, so the model was
# 5 mm short fore-aft and 10 mm TOO WIDE laterally. The width matters more: it
# is the roll axis, where this robot has the least margin and where the stance
# fix had already narrowed things.
#
# The real sole carries a further ~10 mm tab on the width that Steve judged not
# load-bearing; it is deliberately not modelled, so the footprint here is the
# conservative one.
SOLE_WIDTH_V1 = 0.040
SOLE_WIDTH_V2 = 0.030

# The V2 boot weighs 56 g (measured 2026-09-08) against V1's 69 g, so its delta
# over the 18 g standard pad is 38 g, not 51. Robot total with V2 boots: ~866 g.
PAD_MASS_V2 = 0.038


def make_sprung_foot_spec_fn(
    stiffness: float,
    travel: float = TRAVEL,
    damping: float | None = DAMPING,
    h_add: float = H_ADD,
    pad_mass: float = PAD_MASS,
    preload: float = SPRING_PRELOAD,
    damping_ratio: float = DAMPING_RATIO,
    sole_length: float = SOLE_LENGTH_V1,
    sole_width: float = SOLE_WIDTH_V1,
) -> Callable[[], mujoco.MjSpec]:
    """Build a zero-argument ``spec_fn`` for a sprung-foot MicroDuck.

    ``EntityCfg.spec_fn`` must take no arguments, so the spring parameters are
    captured in a closure. ``travel=0.0`` yields the LOCKED control variant:
    identical geometry and mass, no compliance.

    Args:
        stiffness: spring rate in N/m, applied to both feet.
        travel: stroke in m. 0.0 locks the spring.
        damping: absolute N.s/m on the spring DoF. ``None`` (the default)
            derives it from ``damping_ratio`` as ``2*zeta*sqrt(k*pad_mass)``,
            which holds the pad's damping ratio constant as pad_mass varies.
            Pass an explicit value once the prototype's real hysteresis is
            measured.
        damping_ratio: target zeta for the pad-on-spring subsystem, used only
            when ``damping`` is None.
        h_add: metres of height the mechanism adds below the existing sole.
        pad_mass: mass per pad in kg.
        preload: metres of precompression built into the mechanism at
            assembly. Applied as ``springref = -preload`` (see below).
    """

    resolved_damping = (
        damping if damping is not None else damping_for(stiffness, pad_mass, damping_ratio)
    )

    def _spec_fn() -> mujoco.MjSpec:
        # THE ALL-COLLISIONS BASE, not robot_walk.xml. The hop policy uses the
        # head hard -- it is ~35% of body mass and swinging it is worth real
        # height -- and robot_walk.xml has NO head or neck collision geometry
        # at all: only trunk_base, leg and leg_2, on their own
        # contype=2/conaffinity=2 layer. So the head swept straight through the
        # body and the policy was free to exploit a self-intersection that
        # cannot happen on hardware. apirrone's robot_allcollisions.xml
        # (develop 08680d3d) carries 70 collidable geoms, 16 of them on
        # `jaw_soft` and 6 on the neck.
        #
        # Same robot otherwise: both XMLs compile to 737.2 g with 15 joints and
        # 14 actuators, so this changes contact only, not mass or kinematics.
        spec = get_allcollisions_spec()
        for side in ("left", "right"):
            ankle = spec.body(f"ankle_{side}")

            # Retire the rigid sole: rename it and switch off its contact, so
            # the name `{side}_foot_collision` is free for the pad below. Left
            # in place it would keep answering the feet_ground_contact sensor
            # while floating h_add above the ground.
            old_geom = spec.geom(f"{side}_foot_collision")
            old_geom.name = f"{side}_sole_disabled"
            old_geom.contype = 0
            old_geom.conaffinity = 0
            spec.site(f"{side}_foot").name = f"{side}_foot_old"
            # -y is downward in world at the home pose, so a negative y offset
            # puts the pad below the ankle.
            # Centred under the foot SOLE, not under the ankle joint origin.
            # See SOLE_OFFSET_* above for the measurement and what it cost.
            fore_aft = -SOLE_OFFSET_FOREAFT if side == "left" else SOLE_OFFSET_FOREAFT
            pad = ankle.add_body(
                name=f"{side}_foot_pad",
                pos=[fore_aft, -(ANKLE_TO_SOLE + h_add), SOLE_OFFSET_LATERAL],
            )
            # travel == 0.0 is the LOCKED control arm: no joint at all, so the
            # pad is a rigid child of the ankle (identical mass and height,
            # zero DoF). A slide joint with range [0, 0] compiles fine but is
            # NOT locked -- MuJoCo leaves `limited` at AUTO in that case
            # (range == the joint-type default), so the joint is actually
            # unconstrained and held only by the spring. That silently turns
            # the control arm into an infinite-travel spring, which defeats
            # its purpose of isolating "extra height/mass" from "compliance".
            if travel > 0.0:
                joint = pad.add_joint(
                    name=f"passive_{side}_foot_spring",
                    type=mujoco.mjtJoint.mjJNT_SLIDE,
                )
                joint.axis = list(SPRING_AXIS)
                joint.range = [0.0, travel]
                joint.limited = 1
                # These MUST be 3-arrays; MjsJoint rejects a scalar. Only
                # element 0 is used by the compiler.
                joint.stiffness = np.array([stiffness, 0.0, 0.0])
                joint.damping = np.array([resolved_damping, 0.0, 0.0])
                # Stiffen the mechanical end-stop; see the constants above.
                joint.solref_limit = np.array(SPRING_SOLREF_LIMIT)
                joint.solimp_limit = np.array(SPRING_SOLIMP_LIMIT)
                # MuJoCo's spring force is -stiffness * (qpos - springref).
                # Our convention is q=0 extended, q>0 compressed, so a NEGATIVE
                # springref puts a compression-resisting force at q=0: the
                # spring is pressed against its own extension stop, i.e. the
                # preload. Unlike stiffness/damping, springref is a SCALAR on
                # MjsJoint (a 3-array raises TypeError).
                joint.springref = -preload
                # Set explicitly to override the `microduck` childclass joint
                # defaults (0.1 / 0.005) this joint would otherwise inherit.
                # See the constants above. Unlike stiffness/damping these two
                # are SCALARS on MjsJoint (a 3-array raises TypeError).
                joint.frictionloss = SPRING_FRICTIONLOSS
                joint.armature = SPRING_ARMATURE
            # Re-use the ORIGINAL names so the contact sensor, the terrain
            # height-scan frames, foot_clearance and foot_slip all keep working
            # with no config change.
            pad.add_geom(
                spec.find_default(_COLLISION_CLASS),
                name=f"{side}_foot_collision",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                # (fore-aft, thickness, lateral) half-extents; only fore-aft varies
                # between boot versions. NOTE: ANKLE_TO_SOLE cancels the contact-
                # penetration difference between mesh sole and box pad, and
                # penetration depends on contact PRESSURE, so a longer sole sits
                # ~1-2 mm higher than the 30 mm H_ADD target until re-tuned.
                size=[sole_length / 2.0, _PAD_HALF_EXTENTS[1], sole_width / 2.0],
                pos=[0.0, 0.0, 0.0],
                mass=pad_mass,
            )
            pad.add_site(name=f"{side}_foot", pos=[0.0, 0.0, 0.0])

            # VISUAL ONLY: the boot column, so the mechanism can be SEEN.
            #
            # The pad is a 8 mm-thick box floating a few centimetres below the
            # ankle with nothing drawn in between, which makes the viewer
            # genuinely hard to read -- you cannot tell a compressing boot from
            # a retracting leg, and that distinction is the whole question in
            # this campaign. These two geoms draw the telescoping pair: a fixed
            # sleeve on the ankle and a shaft on the pad, so the overlap between
            # them IS the spring travel, visible directly.
            #
            # contype/conaffinity 0 and mass 0, so they are inert: no contacts,
            # no inertia, no effect on the dynamics. `group=2` keeps them on the
            # viewer's visual layer rather than the collision layer.
            # The CONTACT PAD ITSELF is invisible in the viewer: it lives in
            # geom group 3, the collision layer, which the viewer hides by
            # default. So the boot appeared to end in nothing -- "point like".
            # This twin draws the actual 50 x 40 mm sole on the visual layer.
            pad.add_geom(
                name=f"{side}_sole_visual",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[sole_length / 2.0, _PAD_HALF_EXTENTS[1], sole_width / 2.0],
                pos=[0.0, 0.0, 0.0],
                contype=0, conaffinity=0, mass=0.0, group=2,
                rgba=[0.15, 0.15, 0.18, 1.0],
            )
            if travel > 0.0:
                span = ANKLE_TO_SOLE + h_add
                ankle.add_geom(
                    name=f"{side}_boot_sleeve",
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    # -y is downward at the home pose, as for the pad above.
                    # The sleeve is on the ANKLE body, so it needs the same
                    # fore-aft / lateral offsets the pad got -- otherwise it
                    # stays on the ankle centreline and draws as a stray
                    # cylinder 15.7 mm to the side of its own boot.
                    fromto=[fore_aft, -0.004, SOLE_OFFSET_LATERAL,
                            fore_aft, -(span - travel), SOLE_OFFSET_LATERAL],
                    size=[0.008, 0.0, 0.0],
                    contype=0, conaffinity=0, mass=0.0, group=2,
                    rgba=[0.25, 0.25, 0.28, 1.0],
                )
                pad.add_geom(
                    name=f"{side}_boot_shaft",
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    fromto=[0.0, 0.0, 0.0, 0.0, span - travel * 0.5, 0.0],
                    size=[0.005, 0.0, 0.0],
                    contype=0, conaffinity=0, mass=0.0, group=2,
                    rgba=[0.85, 0.55, 0.10, 1.0],
                )
        return spec

    return _spec_fn


def make_sprung_foot_robot_cfg(
    stiffness: float,
    travel: float = TRAVEL,
    damping: float | None = DAMPING,
    h_add: float = H_ADD,
    pad_mass: float = PAD_MASS,
    preload: float = SPRING_PRELOAD,
    sole_length: float = SOLE_LENGTH_V1,
    sole_width: float = SOLE_WIDTH_V1,
) -> EntityCfg:
    """EntityCfg for a sprung-foot MicroDuck, spawned h_add higher.

    The spawn must rise by exactly ``h_add`` or the taller foot starts inside
    the floor.
    """
    init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, h_add),
        joint_pos=dict(HOME_FRAME.joint_pos),
        joint_vel={".*": 0.0},
    )
    return EntityCfg(
        spec_fn=make_sprung_foot_spec_fn(
            stiffness, travel, damping, h_add, pad_mass, preload,
            sole_length=sole_length,
            sole_width=sole_width,
        ),
        init_state=init_state,
        collisions=(FULL_COLLISION,),
        articulation=EntityArticulationInfoCfg(
            actuators=(actuators,),
            soft_joint_pos_limit_factor=0.9,
        ),
    )
