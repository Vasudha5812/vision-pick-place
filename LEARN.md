# How this project works: study guide for interviews

Read this file and the code side by side. For each part, make sure you can answer the questions at the end of the section in your own words.

## 1. Vision (`pickplace/vision.py`)

**Camera image.** `render()` draws the table from above, then warps it with a perspective transform so it looks like a tilted camera took it. Noise and uneven lighting make it realistic.

**Calibration.** A camera gives positions in pixels, but the robot needs metres. There are 4 black markers at known world positions. We find them in the image (dark, round blobs), then `cv2.findHomography` computes a 3×3 matrix **H** that maps any pixel (u, v) to a point (x, y) on the table plane. A homography is enough because all objects lie on one flat plane.

**Detection.** Convert to HSV (Hue, Saturation, Value). Colour lives mainly in Hue, so HSV is far less sensitive to lighting than RGB. For each colour: `cv2.inRange` gives a mask, opening removes noise specks, closing fills holes, `findContours` finds blobs, and image moments give each blob's centre. Red needs two ranges because its hue wraps around 0/180.

*Questions:* Why HSV instead of RGB? Why do you need at least 4 points for a homography? What happens if the table is not flat?

## 2. Kinematics (`pickplace/kinematics.py`)

**Forward kinematics (FK):** angles → position.
`x = L1·cos(q1) + L2·cos(q1+q2)`, `y = L1·sin(q1) + L2·sin(q1+q2)`

**Inverse kinematics (IK):** position → angles, using the law of cosines:
`cos(q2) = (x² + y² − L1² − L2²) / (2·L1·L2)`, then `q1 = atan2(y, x) − atan2(L2·sin q2, L1 + L2·cos q2)`.
There are two solutions (elbow on either side). We choose the one closest to the current angles so the arm doesn't flip.

**Jacobian J:** maps joint velocities to end-effector velocity (`v = J·q̇`). When `det(J) = 0` the arm is at a singularity (fully stretched or folded) and loses a direction of motion.

*Questions:* Why does IK have two solutions? Where are this arm's singularities? Why is IK harder than FK for a 6-DOF robot?

## 3. Trajectory (`pickplace/trajectory.py`)

A **quintic polynomial** moves each joint from start to goal with zero velocity and zero acceleration at both ends, so motion is smooth (no jerks that shake the robot). The duration is the shortest time that keeps peak velocity and acceleration inside the joint limits. Both joints use the same duration so they start and stop together.

*Questions:* Why not just jump straight to the target angle? Why a quintic instead of a cubic?

## 4. Dynamics (`pickplace/dynamics.py`)

Real arms obey `M(q)·q̈ + C(q, q̇)·q̇ + B·q̇ = τ`:
- **M(q)** mass matrix: the inertia changes with the elbow angle (a stretched arm is harder to swing).
- **C** Coriolis and centrifugal effects: moving one joint pushes on the other.
- **B** friction; **τ** motor torques, limited to what the motors can give.

The simulator solves for `q̈` and integrates every 1 ms.

## 5. Control (`pickplace/control.py`)

**PID** on each joint: `τ = Kp·e + Ki·∫e + Kd·ė`, where e is the angle error.
- P pulls toward the target, D adds damping (reduces overshoot), I removes steady-state error.
- Gains come from a chosen natural frequency ωn = 30 rad/s and damping ζ = 1 (critically damped): `Kp = J·ωn²`, `Kd = 2ζωn·J`.
- The integral is clamped (**anti-windup**) so it doesn't build up while the motor is saturated.

**Feedforward:** we already know the planned acceleration, so we compute the torque the model says is needed (`M·q̈_ref + C·q̇ + B·q̇`) and add it. PID then only fixes small errors. Result: tracking error drops by 81%.

*Questions:* What does each PID term do? What is integral windup? Why does feedforward help so much?

## 6. Task logic and optimisation (`pickplace/sim.py`)

For each detected cube: move above it → grasp (succeeds only if within 12 mm) → move to its colour's bin → release. Success is checked against the **true** cube positions, not the detected ones, so vision errors would show up as failures.

**Pick order:** naive (as detected), greedy (nearest next), optimal (try every order; fine for ≤ 7 cubes since 7! = 5040). Optimal saves 8.4% cycle time. For many objects this is a travelling-salesman-type problem, so greedy or heuristics are used instead.

## 7. Talking about it honestly

Say clearly that it's a **simulation**, with the simplifications listed in the README. Then explain what you would do next: port to ROS 2 + Gazebo + MoveIt 2, extend to 6-DOF, test on a real arm. Interviewers respect a candidate who knows the limits of their work.
