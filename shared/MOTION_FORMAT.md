# Hero motion pack — format contract (LabWebsite v1.0)

File: `shared/assets/motion/cloud-hands.bin` (little-endian, no compression). Referenced from
`versions/labwebsite-v1.0/index.html` as `../../shared/assets/motion/cloud-hands.bin`.

Layout:
1. `uint32 headerLen` — byte length of the JSON header, already padded with trailing spaces to a multiple of 4.
2. JSON header (UTF-8), fields:
   - `format`: "hmmh-motion-1"
   - `name`: "Cloud Hands (云手)"; `source`: path of the source stream; `motion`: "generated (procedural), not recorded"
   - `body`: "SMPL-X neutral, quadric-decimated web mesh"; `license_note`: one line
   - `fps` (number), `frames` (F), `vertexCount` (V), `faceCount` (T)
   - `offset` [3], `scale` [3] — decode: `metres[axis] = offset[axis] + u16 * scale[axis]`
   - `up`: "y" (the pack is re-oriented so +Y is up, +Z toward the viewer/front of the body, X to the body's left→right as seen from the front; state it explicitly)
   - `groundY`: lowest foot vertex over all frames (metres) — the viewer puts the floor ellipse here
   - `centre`: [x, z] mean of the pelvis-region vertices at frame 0 (should be ~0)
   - `bounds`: {"min":[3],"max":[3]} over all frames
   - `keyframes`: [0, F/4, F/2, 3F/4] as integers — the 4 markers on the floor ellipse
   - `loop`: "pingpong" or "cycle" (whichever hides the seam better; say why in NOTES.md)
3. `faces`: T × 3 × uint16, triangle-major, counter-clockwise front faces (three.js FrontSide).
4. `vertices`: F × V × 3 × uint16, frame-major (`(frame*V + v)*3 + axis`).

Targets: V ≈ 1,500–2,000, T ≈ 3,000–4,000, F = 48 (15 fps, viewer interpolates), whole file ≤ 600 KB.
Quality: max reconstruction error of the decimated surface vs the full 10,475-vertex surface ≤ 5 mm at every frame (report it).

---

# v2 — skinned sequence pack (supersedes v1 for LabWebsite v1.0)

File: `shared/assets/motion/taiji-sequence.bin` (little-endian). The v1 `cloud-hands.bin` stays on disk as the record.

Layout:
1. `uint32 headerLen` — byte length of the JSON header, padded with trailing spaces to a multiple of 4.
2. JSON header:
   - `format`: "hmmh-motion-2"; `name`: e.g. "Tai Chi sequence (generated)"; `motion`: "generated (procedural), not recorded"
   - `body`, `license_note`, `source`: as in v1
   - `fps` (30), `frames` F, `vertexCount` V, `faceCount` T, `jointCount` J (= 22: SMPL-X pelvis + 21 body joints, in SMPL-X order; finger, jaw and eye joints are folded into wrist / head)
   - `parents`: [J] ints, −1 for the pelvis
   - `restJoints`: [J][3] metres, the rest-pose joint positions in the SAME frame as the rest vertices (+Y up, +Z body front toward the viewer, +X = subject's left)
   - `restOffset`[3], `restScale`[3] — decode the rest vertices: `metres = restOffset + u16 * restScale`
   - `groundY`: lowest foot vertex over all frames of the SKINNED output (metres); `centre` [x,z] of the rest pelvis (≈0)
   - `bounds` {min,max} over all skinned frames
   - `segments`: [{"name": "...", "start": frame, "end": frame}] one per chained action, in play order
   - `keyframes`: [4] frame indices for the 4 floor markers (spread over the sequence)
   - `loop`: "cycle" (the sequence must close on itself: last frame blends into frame 0)
   - `weightsNote`: "uint8, /255, renormalise per vertex"
3. `restVertices`: V × 3 uint16.
4. `faces`: T × 3 uint16, counter-clockwise from outside.
5. `skinIndex`: V × 4 uint8 joint indices; 6. `skinWeight`: V × 4 uint8 (sum ≈ 255; the viewer renormalises).
7. `rotations`: F × J × 4 int16 quaternions (x, y, z, w) × 32767, LOCAL rotation of each joint relative to its parent (joint 0 carries the global orientation). Bone pose = rest translation (restJoints[j] − restJoints[parent]) followed by this rotation, exactly as three.js `Bone` + `SkinnedMesh` expect.
8. `rootTranslation`: F × 3 int16, metres × 1000 (zero if the source is pelvis-centred).

Sections are each padded to a multiple of 4 bytes. Targets: V ≈ 1,500–2,000, T ≈ 3,000–4,000, F ≈ 450–600 (15–20 s at 30 fps), file ≤ 400 KB.
Quality: the skinned LBS output (Python re-implementation of the same formula) vs the exact SMPL-X vertices for the same poses: report max / p99 error in mm per segment; expect a few cm at bent joints (no pose correctives), which is accepted.

---

# v3 note — recorded packs (`taiji-umons-f.bin`, `taiji-umons-m.bin`; LabWebsite v1.1)

Same binary layout and the same v2 fields. Additive header fields; a reader that ignores them works unchanged:
- `citation`: the UMONS-TAICHI reference.
- `groundYNote`. `groundY` is now the **planted-foot level**: the median lowest foot vertex while a foot is in contact. The
  global minimum can be up to 2 cm lower, under the LBS crease of a strongly extended toe; this is reported in NOTES.
- `segments[].trial`: the recorded trial id. A form can appear twice when the loop starts inside it; the last entry then
  continues into frame 0.
- `bands` (male pack only, for Home 01 / B): `{step, units, performers, note, series: {wristHeight|kneeFlexion|pelvisYaw:
  {mean: [ceil(F/step)], sd: [...]}}}`. Values are sampled every `step` frames and are `null` where no recorded band exists
  (the transitions).
- One page can hold several packs. The preload links `motion-src` (hero) and `motion-src-<name>` (e.g. `motion-src-m`,
  card A) become the inline blocks `motion-data` / `motion-data-<name>` in the dist and the artifact (`tools/build.py`
  `inline_motion`).
