"""Abstract Raibert monopod: does a long-stroke pogo give a VISIBLE sustained hop?

The boot answered no, and the reason was stroke: 12 mm gives a 32-46 ms stance,
so even a 40 mm hop reads as a buzz. Flight time is fixed by amplitude alone
(181 ms at 40 mm), so the only lever on frequency is stance, and stance is
bought with stroke. This sizes that.

MODEL -- two prismatic joints IN SERIES, which is the whole point:

    trunk (free joint)
      |-- hip_pitch / hip_roll      aim the leg (Raibert's attitude channel)
          |-- leg_act    (slide, ACTUATED)   <- "the legs", the energy source
              |-- leg_spring (slide, SPRUNG) <- the pogo
                  |-- foot sphere -> ground

The legs act THROUGH the spring, so leg extension does work into it and the
spring returns that work fast. A 12 mm boot could not host this because the
leg's push was over before the boot had moved.

Deliberately NOT mjlab: no 61-D obs contract, no BAM actuators, no DR. This is
a sizing spike, not a policy.

RESULT (f_leg 25 N, the value that reproduces the real robot):

    stroke       k     rise     freq    used  bottom
      12mm    6518    17.6mm   4.48Hz   8.7mm   0.0%   <- the real boot's travel
      20mm    2708    41.3mm   3.44Hz  18.7mm   2.6%
      30mm    1404    53.5mm   3.01Hz  28.5mm   2.9%   <- optimum
      40mm     903    46.5mm   2.97Hz  30.0mm   0.0%
      60mm     501    27.6mm   3.27Hz  30.3mm   0.0%
      80mm     338    18.3mm   3.47Hz  30.4mm   0.0%

The 12 mm row is the CALIBRATION: the real robot, on its real 12 mm boot,
measures 24.0 mm at 4.77 Hz. The model says 17.6 mm at 4.48 Hz -- same
operating point, slightly conservative -- which is what licenses reading the
other rows.

Two things worth knowing before using this:

  * LONGER IS NOT BETTER. Performance peaks near 30-40 mm and falls away
    sharply: 60 mm gives 27.6 mm and 80 mm gives 18.3 mm. Past the optimum the
    spring is too soft to convert its stored energy into a launch within the
    stroke and force the legs have, so the energy goes into travel instead of
    into the body.
  * FRICTION BARELY MATTERS, which contradicts the prediction that led to this
    script. Across 0.2 / 0.65 / 1.5 N of Coulomb loss the 30 mm point moves
    53.5 -> 53.3 -> 53.0 mm, under 1%. Friction dominates a PASSIVE bouncer;
    this one is pumped every cycle, so friction raises the required thrust
    rather than killing the hop.

NOT MODELLED, and each could cost: balance (this is one vertical DoF, and a
point foot under a free body topples -- that was measured, not assumed), the
real leg geometry (no pure prismatic push axis), servo bandwidth, and the pump
phase, which here is detected exactly.
"""
from __future__ import annotations

import argparse
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

G = 9.81
M_TRUNK = 0.79          # MicroDuck minus the 76 g of boots
M_LEG = 0.05            # upper telescoping member
M_FOOT = 0.07           # rod + foot: the mass that must be ACCELERATED each cycle
H_TARGET = 0.040        # the "visible" hop we are sizing for
L_REST = 0.090          # nominal leg length at full extension


def sized_stiffness(stroke: float, m_total: float, h: float = H_TARGET) -> float:
    """0.5*k*x^2 = M*g*(h + x): the spring holds the whole cycle at bottom of stance."""
    return 2.0 * m_total * G * (h + stroke) / stroke ** 2


# What the two legs together can push, in newtons. The measured gait reaches
# 24 mm of rise = 0.20 J; over a ~30 mm leg stroke that is ~7 N of NET upward
# force, so ~16 N total against a 9 N body weight, with peaks higher. 35 N is
# ~4x body weight and matches the peak forces in the boot sizing table. This is
# an assumption, not a measurement, and every number below scales with it.
F_LEG = 35.0


def build(stroke: float, k: float, damping: float, friction: float,
          f_leg: float = F_LEG) -> mujoco.MjModel:
    # VERTICAL-ONLY, DELIBERATELY. The first version gave the trunk a free joint
    # with hip PD to zero and nothing holding the body upright -- a point foot
    # under a free body is an inverted pendulum, so it toppled before the spring
    # ever loaded (measured: 0.0 mm of travel used, trunk drifting down 30-44 mm,
    # contact chattering at 40-1000 Hz). Balance is a real problem but a
    # SEPARATE one; this spike sizes the spring, and the sizing question --
    # amplitude, frequency, sustain -- lives entirely in the vertical axis.
    xml = f"""
<mujoco model="monopod_vertical">
  <option timestep="0.0002" integrator="implicitfast"/>
  <default>
    <geom contype="1" conaffinity="1" friction="1.0 0.005 0.0001"
          solref="0.004 1" solimp="0.95 0.99 0.001"/>
  </default>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1" rgba=".3 .3 .3 1"/>
    <body name="trunk" pos="0 0 {L_REST + stroke + 0.02}">
      <joint name="slide_z" type="slide" axis="0 0 1" limited="false"/>
      <geom name="trunk_g" type="box" size="0.045 0.035 0.050" mass="{M_TRUNK}" rgba=".8 .3 .2 1"/>
      <body name="leg_mid" pos="0 0 -0.050">
        <!-- THE LEGS. Actuated: the only place energy enters the system. -->
        <joint name="leg_act" type="slide" axis="0 0 1" range="-0.035 0.0" damping="0.05"/>
        <geom name="leg_mid_g" type="capsule" fromto="0 0 0 0 0 -0.045" size="0.007"
              mass="{M_LEG}" rgba=".2 .4 .8 1"/>
        <body name="leg_low" pos="0 0 -0.045">
          <!-- THE POGO. Passive, sprung, hard stop at full stroke. -->
          <!-- COMPRESSION IS POSITIVE q: leg_low hangs at -z from leg_mid and the
               axis is +z, so sliding UP shortens the leg. The range was written
               as [-stroke, 0] at first, which permits only EXTENSION -- the
               spring was forbidden from compressing and sat at 0.12 mm under a
               9 N load while the body rested on the contact instead. -->
          <joint name="leg_spring" type="slide" axis="0 0 1" range="0.0 {stroke}"
                 stiffness="{k}" damping="{damping}" frictionloss="{friction}"
                 springref="0" limited="true" solreflimit="0.002 1"/>
          <geom name="foot_g" type="sphere" size="0.012" pos="0 0 -0.025"
                mass="{M_FOOT}" rgba=".9 .9 .2 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <!-- THE ACTUATOR MUST BE STIFF, or it becomes the softest spring in the
         chain and the pogo never sees the load. At kp=900 under a 9 N robot it
         sagged 20 mm while the 903 N/m pogo compressed 0.12 mm: all the
         deflection went into the "rigid" member and the sweep measured
         nothing. kp=50000 gives 0.18 mm under the same load, so the series
         ordering is what the drawing says: legs rigid, spring compliant. -->
    <!-- FORCERANGE IS THE WHOLE ANSWER. Unlimited, kp=50000 with a 20 mm
         command delivers ~1000 N (100x body weight) and the model cheerfully
         reports 700 mm hops on a 250 mm robot. The sizing question is only
         meaningful against what the real legs can push, so this is the number
         to argue about -- see F_LEG. -->
    <position name="a_leg" joint="leg_act" kp="50000" kv="300" ctrlrange="-0.035 0.0"
              forcerange="-{f_leg} {f_leg}"/>
  </actuator>
</mujoco>
"""
    return mujoco.MjModel.from_xml_string(xml)


def run(stroke, k, thrust, damping=0.4, friction=0.2, seconds=6.0, settle=1.0,
        f_leg=F_LEG):
    """Hop with a bang-bang pump: extend while loaded, retract while airborne."""
    model = build(stroke, k, damping, friction, f_leg)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    jid_spring = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "leg_spring")
    qadr_spring = model.jnt_qposadr[jid_spring]
    vadr_spring = model.jnt_dofadr[jid_spring]
    gid_foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot_g")
    bid_trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")

    dt = model.opt.timestep
    n = int(seconds / dt)
    n_settle = int(settle / dt)

    apexes, contacts, flights, comps, rises = [], [], [], [], []
    air_t = gnd_t = 0.0
    z_peak = -1e9
    z_takeoff = 0.0
    was_air = False
    bottomed = 0

    for i in range(n):
        # contact: is the foot touching the ground?
        touching = any(
            (model.geom_bodyid[c.geom1] == 0 or model.geom_bodyid[c.geom2] == 0)
            and (c.geom1 == gid_foot or c.geom2 == gid_foot)
            for c in data.contact[: data.ncon]
        )
        # PUMP TIMING IS THE WHOLE CONTROLLER. Pushing on CONTACT fights the
        # incoming compression and bleeds energy -- measured: apexes drifting
        # -26 mm over the run, i.e. the hop dying. Raibert thrusts during
        # DECOMPRESSION: wait for the spring to bottom out (its velocity
        # crossing from compressing to extending), then extend into the
        # rebound, so the actuator's work adds to the spring's release instead
        # of cancelling it.
        # q>0 is compression, so the rebound (decompression) is qvel < 0.
        v_spring = float(data.qvel[vadr_spring])
        decompressing = touching and v_spring < 0.0
        data.ctrl[0] = -thrust if decompressing else 0.0

        mujoco.mj_step(model, data)

        q = float(data.qpos[qadr_spring])
        z = float(data.xipos[bid_trunk, 2])
        if i >= n_settle:
            comps.append(q)
            if q >= stroke * 0.995:
                bottomed += 1
            if touching:
                if was_air:
                    flights.append(air_t)
                    apexes.append(z_peak)
                    rises.append(z_peak - z_takeoff)
                    air_t = 0.0
                gnd_t += dt
            else:
                if not was_air:
                    contacts.append(gnd_t)
                    gnd_t = 0.0
                    z_peak = z
                    z_takeoff = z
                air_t += dt
                z_peak = max(z_peak, z)
        was_air = not touching

    comps = np.array(comps)
    # steady state = the last half of the measured window
    def tail(a):
        a = np.array(a)
        return a[len(a) // 2:] if len(a) > 3 else a

    f, c, ap, ri = tail(flights), tail(contacts), tail(apexes), tail(rises)
    if len(f) < 2 or len(c) < 2 or len(ri) < 2:
        return None
    period = f.mean() + c.mean()
    rise = float(ri.mean())                # measured apex minus takeoff height
    return dict(stroke=stroke, k=k, thrust=thrust,
                flight=f.mean(), contact=c.mean(), period=period, freq=1 / period,
                duty=f.mean() / period, rise=rise,
                comp_p95=float(np.percentile(comps, 95)),
                bottomed=bottomed / max(len(comps), 1),
                n_hops=len(f), apex_drift=float(ap[-1] - ap[0]) if len(ap) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thrust", type=float, default=0.020,
                    help="metres of actuator push while loaded = the energy input")
    ap.add_argument("--friction", type=float, default=0.2, help="N of Coulomb loss in the slide")
    ap.add_argument("--f-leg", type=float, default=F_LEG, help="N the legs can push")
    args = ap.parse_args()

    m_total = M_TRUNK + M_LEG + M_FOOT + 0.01
    print(f"total mass {m_total*1000:.0f} g,  target rise {H_TARGET*1000:.0f} mm,  "
          f"thrust {args.thrust*1000:.0f} mm,  slide friction {args.friction} N,  "
          f"leg force limit {args.f_leg:.0f} N ({args.f_leg/(m_total*G):.1f}x body weight)")
    print(f"flight time for a {H_TARGET*1000:.0f} mm hop is {2*math.sqrt(2*H_TARGET/G)*1000:.0f} ms "
          f"regardless of spring\n")
    print(f"{'stroke':>7} {'k':>7} {'rise':>8} {'freq':>8} {'stance':>8} {'flight':>8} "
          f"{'duty':>6} {'used':>8} {'bottom':>7} {'drift':>8}")
    print("-" * 88)
    # 12 mm is the REAL BOOT's travel: it is the calibration row. If the model
    # does not reproduce the measured 24 mm hop there, nothing in the other rows
    # is worth reading.
    for stroke in (0.012, 0.020, 0.030, 0.040, 0.060, 0.080):
        k = sized_stiffness(stroke, m_total)
        r = run(stroke, k, args.thrust, friction=args.friction, f_leg=args.f_leg)
        if r is None:
            print(f"{stroke*1000:>6.0f}mm {k:>7.0f}   -- no sustained hopping --")
            continue
        print(f"{stroke*1000:>6.0f}mm {k:>7.0f} {r['rise']*1000:>7.1f}mm {r['freq']:>7.2f}Hz "
              f"{r['contact']*1000:>7.0f}ms {r['flight']*1000:>7.0f}ms {r['duty']*100:>5.0f}% "
              f"{r['comp_p95']*1000:>6.1f}mm {r['bottomed']*100:>6.1f}% "
              f"{r['apex_drift']*1000:>+7.1f}mm")
    print("\nMicroDuck today, for scale:  24.0 mm rise, 4.77 Hz, 46 ms stance, 78% duty")


if __name__ == "__main__":
    main()
