#!/usr/bin/env python3
"""umons_motion.py -- real Tai Chi motion from UMONS-TAICHI (Tits et al. 2018, Data in Brief 19, doi:10.1016/j.dib.2018.05.088;
CC BY-NC-SA 4.0) fitted to SMPL-X, for the hero motion pack v3 (tools/motion_pack.py build3 writes the pack).

Data (Zenodo record 2784581): Segmented_Kinect.zip (Kinect v2, 25 joints, 30.3 Hz) and single members of Segmented_TSV.zip
(Qualisys, 68 markers, 179 Hz) read by HTTP range requests. Both share trial ids PxxTxxCxxGxxDxxSxx and are synchronous.

Subcommands (run with smplx_meridian_project/.venv; torch is imported only in `fit`, never together with open3d):
  fit       one trial -> SMPL-X local rotations [n,22,4] at 30 fps + root path + contact flags (npz), from --source kinect|qualisys
Library functions are imported by motion_pack.py build3 (sequence assembly, root/contact solve, leg IK).
Frames: world z-up (as stored) -> canonical SMPL-X frame (+Y up, +Z body front, +X subject's left): (x,y,z) = (Yw, Zw, Xw),
then a yaw about +Y so the trial's mean facing is +Z."""
import argparse
import json
import os
import sys

import numpy as np

MODELS = "/Users/elm/Documents/GitHub/smplx_meridian_project/models/smplx"
KIN = ["SpineBase", "SpineMid", "Neck", "Head", "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft", "ShoulderRight",
       "ElbowRight", "WristRight", "HandRight", "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft", "HipRight", "KneeRight",
       "AnkleRight", "FootRight", "SpineShoulder", "HandTipLeft", "ThumbLeft", "HandTipRight", "ThumbRight"]
KJ = {n: i for i, n in enumerate(KIN)}
FPS = 30.0
J22 = 22


# ------------------------------------------------------------------------------------------------ model skeleton
def load_skeleton(gender="NEUTRAL"):
    """Rest joints (pelvis at the origin) of the SMPL-X model (betas 0) + the fixed points the fit needs."""
    np.seterr(all="ignore")
    M = np.load(os.path.join(MODELS, f"SMPLX_{gender}.npz"), allow_pickle=True)
    vt = M["v_template"].astype(np.float64)
    J = M["J_regressor"].astype(np.float64) @ vt
    J0 = J[0].copy(); J -= J0; vt = vt - J0
    par = M["kintree_table"][0].astype(np.int64); par[0] = -1
    lab = M["weights"].argmax(1)
    sk = {"J": J[:J22], "parents": par[:J22], "gender": gender}
    # hand centre = mean of the index/middle/ring/pinky MCP joints (fixed in the wrist frame: fingers are folded into the wrist)
    sk["handL"] = J[[25, 28, 31, 34]].mean(0) - J[20]; sk["handR"] = J[[40, 43, 46, 49]].mean(0) - J[21]
    sk["headc"] = vt[np.isin(lab, [15, 22, 23, 24])].mean(0) - J[15]
    # sole points (heel, ball, toe) per foot: lowest rest vertices of the foot, used for floor contact
    for s, ja, jf in (("L", 7, 10), ("R", 8, 11)):
        v = vt[np.isin(lab, [ja, jf])]
        lo = v[:, 1] < v[:, 1].min() + 0.012
        z0, z1 = v[:, 2].min(), v[:, 2].max(); rear = v[v[:, 2] < z0 + 0.25 * (z1 - z0)]
        heel = rear[np.argmin(rear[:, 1])].copy()                 # on the mesh, under the calcaneus (not the rounded back)
        toe = v[lo][np.argmax(v[lo][:, 2])].copy()
        heel[1] = toe[1] = v[:, 1].min()                         # both contact points on the sole plane of the rest foot
        sk["heel" + s] = heel - J[ja]; sk["toe" + s] = toe - J[jf]
    sk["sole_y"] = float(vt[np.isin(lab, [7, 8, 10, 11]), 1].min())
    sk["leg"] = float(np.linalg.norm(J[4] - J[1]) + np.linalg.norm(J[7] - J[4]))
    return sk


# ------------------------------------------------------------------------------------------------ data
def to_canonical(P):
    """world z-up [...,3] -> canonical (Yw, Zw, Xw)"""
    return P[..., [1, 2, 0]]


def yaw_rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def resample(t, X, n=None, fps=FPS):
    """linear resampling of [m,...] samples at times t (s) to a uniform fps grid from t[0]"""
    T = t[-1] - t[0]
    n = n or int(np.floor(T * fps)) + 1
    g = t[0] + np.arange(n) / fps
    flat = X.reshape(len(X), -1)
    out = np.stack([np.interp(g, t, flat[:, k]) for k in range(flat.shape[1])], 1)
    return g - t[0], out.reshape((n,) + X.shape[1:])


def despike(X, thr=0.05, sigma=4):
    """per joint: samples farther than thr (m) from a heavily smoothed copy are treated as missing and re-interpolated"""
    from scipy.ndimage import gaussian_filter1d, median_filter
    Y = median_filter(X, size=(5, 1, 1), mode="nearest")
    S = gaussian_filter1d(Y, sigma, axis=0, mode="nearest")
    bad = np.linalg.norm(X - S, axis=-1) > thr
    out = X.copy(); t = np.arange(len(X))
    for j in range(X.shape[1]):
        b = bad[:, j]
        if b.any() and (~b).sum() > 2:
            for k in range(3):
                out[b, j, k] = np.interp(t[b], t[~b], X[~b, j, k])
    return out, bad


def load_kinect(cache, tid):
    z = np.load(cache)
    ids = list(z["trial_id"]); i = ids.index(tid)
    s, e = z["offsets"][i], z["offsets"][i + 1]
    return z["t"][s:e].astype(np.float64), z["x"][s:e].astype(np.float64)


def kinect_targets(t, X, sigma):
    """Kinect joints (world, m) -> 30 fps canonical joint positions and quality numbers."""
    from scipy.ndimage import gaussian_filter1d
    tt, P = resample(t, X)
    P = to_canonical(P)
    P, bad = despike(P)
    if sigma > 0:
        P = gaussian_filter1d(P, sigma, axis=0, mode="nearest")
    L = {n: P[:, KJ[n]] for n in KIN}
    q = {"despiked_fraction": {n: float(bad[:, KJ[n]].mean()) for n in KIN}}
    return tt, L, q


def load_qualisys(path):
    z = np.load(path)
    return float(z["fps"]), [str(m) for m in z["markers"]], z["x"].astype(np.float64)


def qualisys_landmarks(fps_src, names, X, sigma):
    """Markers (world z-up, m, 179 Hz) -> anatomical joint centres / axes at 30 fps in the canonical frame."""
    from scipy.ndimage import gaussian_filter1d
    M = {n: i for i, n in enumerate(names)}
    X = gaussian_filter1d(X, 0.5 * fps_src / FPS, axis=0, mode="nearest")          # anti-alias before 30 fps
    t = np.arange(len(X)) / fps_src
    tt, X = resample(t, X)
    X = to_canonical(X)
    if sigma > 0:
        X = gaussian_filter1d(X, sigma, axis=0, mode="nearest")
    m = lambda n: X[:, M[n]]
    nz = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    L = {}
    # pelvis (ISB): lateral from the ASIS, forward in the ASIS / mid-PSIS plane; Harrington 2007 hip joint centres
    asis = (m("L_IAS") + m("R_IAS")) / 2; psis = (m("L_IPS") + m("R_IPS")) / 2
    lat = nz(m("L_IAS") - m("R_IAS"))                                              # toward the subject's left
    fw = asis - psis; fw = nz(fw - np.sum(fw * lat, 1, keepdims=True) * lat)
    up = np.cross(fw, lat)                                                         # forward x left = up
    PW = np.median(np.linalg.norm(m("L_IAS") - m("R_IAS"), axis=1)); PD = np.median(np.linalg.norm(asis - psis, axis=1))
    for s, sg in (("L", 1), ("R", -1)):
        off_f = -0.24 * PD - 0.0099; off_u = -0.30 * PW - 0.0109; off_l = sg * (0.33 * PW + 0.0073)
        L["hip" + s] = asis + off_f * fw + off_u * up + off_l * lat
    L["pelvis_lat"], L["pelvis_fwd"], L["pelvis_up"] = lat, fw, up
    for s in "LR":
        p = s + "_"
        L["knee" + s] = (m(p + "FLE") + m(p + "FME")) / 2
        L["ankle" + s] = (m(p + "FAL") + m(p + "TAM")) / 2
        L["heel" + s] = m(p + "FCC"); L["toe" + s] = (m(p + "FM1") + m(p + "FM5")) / 2
        sgn = 1 if s == "L" else -1
        L["kneeax" + s] = sgn * (m(p + "FLE") - m(p + "FME"))                     # toward the subject's left
        L["footlat" + s] = sgn * (m(p + "FM5") - m(p + "FM1"))                    # toward the subject's left
        L["elbow" + s] = (m(p + "HLE") + m(p + "HME")) / 2
        L["elbowax" + s] = m(p + "HLE") - m(p + "HME")                           # lateral epicondyle (= +Y in the T-pose)
        L["wrist" + s] = (m(p + "RSP") + m(p + "USP")) / 2
        L["forearmax" + s] = m(p + "RSP") - m(p + "USP")                         # radial side (= +Z in the T-pose)
        L["hand" + s] = (m(p + "HM1") + m(p + "HL5")) / 2
        L["handax" + s] = m(p + "HM1") - m(p + "HL5")
    # thorax (ISB-like): up from mid(TV10, STRN) to mid(CV7, CLAV); forward from mid(CV7, TV10) to mid(CLAV, STRN)
    tu = nz((m("CV7") + m("CLAV")) / 2 - (m("TV10") + m("STRN")) / 2)
    tf = (m("CLAV") + m("STRN")) / 2 - (m("CV7") + m("TV10")) / 2; tf = nz(tf - np.sum(tf * tu, 1, keepdims=True) * tu)
    L["thorax_up"], L["thorax_fwd"] = tu, tf
    L["neckbase"] = (m("CV7") + m("CLAV")) / 2
    # head: centre and frame from the four head markers
    hf = (m("LFHD") + m("RFHD")) / 2; hb = (m("LBHD") + m("RBHD")) / 2
    hl = (m("LFHD") + m("LBHD")) / 2; hr = (m("RFHD") + m("RBHD")) / 2
    L["headc"] = (hf + hb) / 2
    hlat = nz(hl - hr); hfw = hf - hb; hfw = nz(hfw - np.sum(hfw * hlat, 1, keepdims=True) * hlat)
    L["head_lat"], L["head_fwd"] = hlat, hfw
    # glenohumeral centre: acromion + constant offset in the thorax frame, chosen so the upper arm length is most constant
    from scipy.optimize import least_squares
    tl = np.cross(tu, tf)                                                          # up x forward = left
    Rth = np.stack([tl, tu, tf], -1)                                               # columns: left, up, forward
    for s, ac in (("L", "LAC"), ("R", "RAC")):
        A = m(ac); E = L["elbow" + s]
        def res(o):
            G = A + np.einsum("nab,b->na", Rth, o)
            d = np.linalg.norm(E - G, axis=1)
            return np.concatenate([d - d.mean(), 0.3 * (o - np.array([0.0, -0.05, 0.0]))])
        o = least_squares(res, np.array([0.0, -0.05, 0.0])).x
        L["shoulder" + s] = A + np.einsum("nab,b->na", Rth, o); L["gh_offset_" + s] = o
    L["pelvis"] = (L["hipL"] + L["hipR"]) / 2
    q = {"PW_m": float(PW), "PD_m": float(PD), "gh_offset_L": L["gh_offset_L"].tolist(), "gh_offset_R": L["gh_offset_R"].tolist()}
    return tt, L, q


def segment_lengths(L, src):
    """per-frame limb segment lengths from the data landmarks (quality check)."""
    if src == "kinect":
        pairs = {"thighL": ("HipLeft", "KneeLeft"), "shankL": ("KneeLeft", "AnkleLeft"), "thighR": ("HipRight", "KneeRight"),
                 "shankR": ("KneeRight", "AnkleRight"), "uarmL": ("ShoulderLeft", "ElbowLeft"), "farmL": ("ElbowLeft", "WristLeft"),
                 "uarmR": ("ShoulderRight", "ElbowRight"), "farmR": ("ElbowRight", "WristRight")}
    else:
        pairs = {"thighL": ("hipL", "kneeL"), "shankL": ("kneeL", "ankleL"), "thighR": ("hipR", "kneeR"), "shankR": ("kneeR", "ankleR"),
                 "uarmL": ("shoulderL", "elbowL"), "farmL": ("elbowL", "wristL"), "uarmR": ("shoulderR", "elbowR"), "farmR": ("elbowR", "wristR")}
    return {k: np.linalg.norm(L[a] - L[b], axis=1) for k, (a, b) in pairs.items()}


def quality(L, src):
    """pre-registered rejection numbers: max joint-centre step per 30 fps frame, (p95-p5)/median of each limb segment"""
    keys = (["SpineBase", "SpineShoulder", "Head", "ShoulderLeft", "ElbowLeft", "WristLeft", "ShoulderRight", "ElbowRight", "WristRight",
             "HipLeft", "KneeLeft", "AnkleLeft", "HipRight", "KneeRight", "AnkleRight"] if src == "kinect" else
            ["pelvis", "neckbase", "headc", "shoulderL", "elbowL", "wristL", "shoulderR", "elbowR", "wristR", "hipL", "kneeL", "ankleL",
             "hipR", "kneeR", "ankleR"])
    step = max(float(np.linalg.norm(np.diff(L[k], axis=0), axis=1).max()) for k in keys)
    var = {k: float((np.percentile(v, 95) - np.percentile(v, 5)) / np.median(v)) for k, v in segment_lengths(L, src).items()}
    return {"max_step_m": step, "segment_variation": var, "worst_segment_variation": max(var.values()),
            "reject": bool(step > 0.08 or max(var.values()) > 0.15)}


def facing_yaw(L, src, frames=None):
    """mean facing angle (rad, about +Y, 0 = +Z) from the hip and shoulder lines"""
    if src == "kinect":
        lat = (L["ShoulderLeft"] - L["ShoulderRight"]) + (L["HipLeft"] - L["HipRight"])
    else:
        lat = (L["shoulderL"] - L["shoulderR"]) + (L["hipL"] - L["hipR"])
    lat = lat if frames is None else lat[frames]
    v = lat.mean(0); fwd = np.cross(v, [0, 1.0, 0])                               # left x up = forward
    return float(np.arctan2(fwd[0], fwd[2]))


def rotate_landmarks(L, R, origin):
    out = {}
    for k, v in L.items():
        if k.startswith("gh_offset"):
            out[k] = v; continue
        if v.ndim == 2 and v.shape[1] == 3:
            pos = not any(k.startswith(a) for a in ("pelvis_", "thorax_", "head_", "kneeax", "footlat", "elbowax", "forearmax", "handax"))
            out[k] = ((v - origin) if pos else v) @ R.T
        else:
            out[k] = v
    return out


# ------------------------------------------------------------------------------------------------ standing calibration
def rodrigues_rot(v, axis, theta):
    """rotate vectors v [n,3] about unit axes [n,3] by angle theta (rad, scalar)"""
    k = axis / np.linalg.norm(axis, axis=1, keepdims=True)
    c, s_ = np.cos(theta), np.sin(theta)
    return v * c + np.cross(k, v) * s_ + k * np.sum(k * v, 1, keepdims=True) * (1 - c)


def sagittal_angle(v, left):
    """alpha = atan2(v . f, v . up) (deg) with up = +Y and f = horizontal forward = left x up; a rotation about `left` by
    +theta adds theta to alpha"""
    up = np.array([0, 1.0, 0]); f = np.cross(left, up); f[:, 1] = 0; f /= np.linalg.norm(f, axis=1, keepdims=True)
    return np.degrees(np.arctan2(np.sum(v * f, 1), v @ up))


CAL_Q = {"pelvis_fwd": "pelvis_lat", "thorax_up": "thorax_lat", "thorax_fwd": "thorax_lat", "head_fwd": "head_lat",
         "trunk": "pelvis_lat", "heeltoeL": "footlatL", "heeltoeR": "footlatR"}
CAL_K = {"trunk": "lat", "head": "lat", "lowspine": "lat"}


def model_rest_alpha(sk):
    J = sk["J"]; X = np.array([[1.0, 0, 0]])
    a = lambda v: float(sagittal_angle(np.asarray(v, float)[None], X)[0])
    return {"pelvis_fwd": 90.0, "thorax_up": 0.0, "thorax_fwd": 90.0, "head_fwd": 90.0, "heeltoeL": 90.0, "heeltoeR": 90.0,
            "trunk": a(J[12] - J[0]), "k_trunk": a(J[12] - J[0]), "k_lowspine": a(J[6] - J[0]),
            "k_head": a(J[15] + sk["headc"] - J[12])}


def qualisys_vectors(L):
    """the sagittally calibrated vectors (and their lateral axes) of a Qualisys landmark dict"""
    V = {"pelvis_fwd": L["pelvis_fwd"], "thorax_up": L["thorax_up"], "thorax_fwd": L["thorax_fwd"], "head_fwd": L["head_fwd"],
         "trunk": L["neckbase"] - L["pelvis"], "heeltoeL": L["toeL"] - L["heelL"], "heeltoeR": L["toeR"] - L["heelR"]}
    A = {"pelvis_lat": L["pelvis_lat"], "thorax_lat": np.cross(L["thorax_up"], L["thorax_fwd"]), "head_lat": L["head_lat"],
         "footlatL": L["footlatL"], "footlatR": L["footlatR"]}
    return V, A


def kinect_vectors(L):
    lat = (L["ShoulderLeft"] - L["ShoulderRight"]) + (L["HipLeft"] - L["HipRight"])
    V = {"trunk": L["SpineShoulder"] - L["SpineBase"], "head": L["Head"] - L["SpineShoulder"], "lowspine": L["SpineMid"] - L["SpineBase"]}
    return V, {"lat": lat}


def standing_alphas(V, A, table):
    return {k: float(np.median(sagittal_angle(V[k], A[table[k]]))) for k in table}


def apply_calibration(L, src, cal, sk):
    """rotate the calibrated vectors about their segment's lateral axis so that the participant's quiet standing
    (G01, Wuji) maps to the SMPL-X rest pose; adds L['cal_<name>'] vectors used by the fit."""
    rest = model_rest_alpha(sk)
    if src == "qualisys":
        V, A = qualisys_vectors(L); table = CAL_Q; pre = ""
    else:
        V, A = kinect_vectors(L); table = CAL_K; pre = "k_"
    used = {}
    for k, axk in table.items():
        th = rest[pre + k] - cal[k]
        L["cal_" + k] = rodrigues_rot(V[k], A[axk], np.radians(th)); used[k] = round(th, 2)
    return used


def calibrate_participant(src, trials, loader):
    """median standing sagittal angles over the participant's G01 (Wuji) trials"""
    out = []
    for tid in trials:
        L = loader(tid)
        V, A = qualisys_vectors(L) if src == "qualisys" else kinect_vectors(L)
        out.append(standing_alphas(V, A, CAL_Q if src == "qualisys" else CAL_K))
    return {k: float(np.median([o[k] for o in out])) for k in out[0]}, out


# ------------------------------------------------------------------------------------------------ the SMPL-X fit (torch)
def fit_trial(L, src, sk, iters=700, w_smooth=4.0, log=None):
    """Whole-trial fit of 22 SMPL-X rotations (6D) to bone directions, segment axes and frames. Returns local rotation
    matrices [n,22,3,3] and per-term residual angles (deg)."""
    import torch
    torch.set_num_threads(4)
    dt = torch.float64
    J = torch.tensor(sk["J"], dtype=dt); par = sk["parents"]
    n = len(next(v for k, v in L.items() if not k.startswith("gh_") and np.ndim(v) == 2))
    T = lambda a: torch.tensor(np.asarray(a), dtype=dt)
    nz = lambda v: v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-9)

    def six_to_mat(x):
        a, b = x[..., :3], x[..., 3:]
        e1 = nz(a); e2 = nz(b - (e1 * b).sum(-1, keepdim=True) * e1); e3 = torch.cross(e1, e2, dim=-1)
        return torch.stack([e1, e2, e3], -1)

    def fk(R):
        G = [None] * J22; P = [None] * J22
        for j in range(J22):
            p = par[j]
            if p < 0:
                G[j] = R[:, j]; P[j] = torch.zeros(n, 3, dtype=dt)
            else:
                G[j] = G[p] @ R[:, j]; P[j] = P[p] + (G[p] @ (J[j] - J[p]))
        return torch.stack(G, 1), torch.stack(P, 1)

    ex, ey, ez = T([1, 0, 0]), T([0, 1, 0]), T([0, 0, 1])
    terms = []                      # (name, weight, fn(G,P) -> model vec [n,3], data vec [n,3], twist_about(G,P) or None)
    def D(a, b): return nz(T(L[b] - L[a]))
    def jd(i, k): return lambda G, P: P[:, k] - P[:, i]
    def ax(j, e): return lambda G, P: G[:, j] @ e
    def pt(j, off): return lambda G, P: G[:, j] @ T(off)
    if src == "kinect":
        K = lambda a, b: nz(T(L[b] - L[a]))
        terms += [("thighL", 1.0, jd(1, 4), K("HipLeft", "KneeLeft"), None), ("thighR", 1.0, jd(2, 5), K("HipRight", "KneeRight"), None),
                  ("shankL", 1.0, jd(4, 7), K("KneeLeft", "AnkleLeft"), None), ("shankR", 1.0, jd(5, 8), K("KneeRight", "AnkleRight"), None),
                  ("footL", 0.3, jd(7, 10), K("AnkleLeft", "FootLeft"), None), ("footR", 0.3, jd(8, 11), K("AnkleRight", "FootRight"), None),
                  ("uarmL", 1.0, jd(16, 18), K("ShoulderLeft", "ElbowLeft"), None), ("uarmR", 1.0, jd(17, 19), K("ShoulderRight", "ElbowRight"), None),
                  ("farmL", 1.0, jd(18, 20), K("ElbowLeft", "WristLeft"), None), ("farmR", 1.0, jd(19, 21), K("ElbowRight", "WristRight"), None),
                  ("handL", 0.3, pt(20, sk["handL"]), K("WristLeft", "HandLeft"), None), ("handR", 0.3, pt(21, sk["handR"]), K("WristRight", "HandRight"), None),
                  ("hipline", 1.0, jd(2, 1), K("HipRight", "HipLeft"), None),
                  ("shoulderline", 1.0, jd(17, 16), K("ShoulderRight", "ShoulderLeft"), None),
                  ("trunk", 1.0, jd(0, 12), nz(T(L["cal_trunk"])), None),
                  ("lowspine", 0.3, jd(0, 6), nz(T(L["cal_lowspine"])), None),
                  ("collarL", 0.5, jd(12, 16), K("SpineShoulder", "ShoulderLeft"), None),
                  ("collarR", 0.5, jd(12, 17), K("SpineShoulder", "ShoulderRight"), None),
                  ("head", 0.5, lambda G, P: P[:, 15] + G[:, 15] @ T(sk["headc"]) - P[:, 12], nz(T(L["cal_head"])), None)]
        robust = 0.15
    else:
        terms += [("thighL", 1.0, jd(1, 4), D("hipL", "kneeL"), None), ("thighR", 1.0, jd(2, 5), D("hipR", "kneeR"), None),
                  ("shankL", 1.0, jd(4, 7), D("kneeL", "ankleL"), None), ("shankR", 1.0, jd(5, 8), D("kneeR", "ankleR"), None),
                  ("uarmL", 1.0, jd(16, 18), D("shoulderL", "elbowL"), None), ("uarmR", 1.0, jd(17, 19), D("shoulderR", "elbowR"), None),
                  ("farmL", 1.0, jd(18, 20), D("elbowL", "wristL"), None), ("farmR", 1.0, jd(19, 21), D("elbowR", "wristR"), None),
                  ("handL", 0.5, pt(20, sk["handL"]), D("wristL", "handL"), None), ("handR", 0.5, pt(21, sk["handR"]), D("wristR", "handR"), None),
                  ("hipline", 1.0, jd(2, 1), D("hipR", "hipL"), None),
                  ("trunk", 1.0, jd(0, 12), nz(T(L["cal_trunk"])), None),
                  ("collarL", 0.5, jd(12, 16), D("neckbase", "shoulderL"), None), ("collarR", 0.5, jd(12, 17), D("neckbase", "shoulderR"), None),
                  # axes: compared after removing the component along the bone (twist only)
                  ("kneeaxL", 0.4, ax(1, ex), nz(T(L["kneeaxL"])), jd(1, 4)), ("kneeaxR", 0.4, ax(2, ex), nz(T(L["kneeaxR"])), jd(2, 5)),
                  ("elbowaxL", 0.4, ax(16, ey), nz(T(L["elbowaxL"])), jd(16, 18)), ("elbowaxR", 0.4, ax(17, ey), nz(T(L["elbowaxR"])), jd(17, 19)),
                  ("forearmaxL", 0.4, ax(18, ez), nz(T(L["forearmaxL"])), jd(18, 20)), ("forearmaxR", 0.4, ax(19, ez), nz(T(L["forearmaxR"])), jd(19, 21)),
                  ("handaxL", 0.3, ax(20, ez), nz(T(L["handaxL"])), pt(20, sk["handL"])), ("handaxR", 0.3, ax(21, ez), nz(T(L["handaxR"])), pt(21, sk["handR"])),
                  # foot frame (heel -> toe forward, metatarsal line lateral); toes (joints 10/11) are held by the prior
                  ("footfwdL", 0.6, ax(7, ez), nz(T(L["cal_heeltoeL"])), None), ("footfwdR", 0.6, ax(8, ez), nz(T(L["cal_heeltoeR"])), None),
                  ("footlatL", 0.3, ax(7, ex), nz(T(L["footlatL"])), ax(7, ez)), ("footlatR", 0.3, ax(8, ex), nz(T(L["footlatR"])), ax(8, ez)),
                  # segment frames
                  ("pelvis_fwd", 0.5, ax(0, ez), nz(T(L["cal_pelvis_fwd"])), None),
                  ("thorax_up", 0.5, ax(9, ey), nz(T(L["cal_thorax_up"])), None), ("thorax_fwd", 0.5, ax(9, ez), nz(T(L["cal_thorax_fwd"])), None),
                  ("head_fwd", 0.5, ax(15, ez), nz(T(L["cal_head_fwd"])), None), ("head_lat", 0.3, ax(15, ex), nz(T(L["head_lat"])), None)]
        robust = 0.3
    # initial pose: identity, pelvis yawed to the data facing; arms rotated down (T-pose -> about 60 deg abduction)
    from scipy.spatial.transform import Rotation as Rot
    R0 = np.tile(np.eye(3), (n, J22, 1, 1))
    R0[:, 16] = Rot.from_euler("z", -60, degrees=True).as_matrix(); R0[:, 17] = Rot.from_euler("z", 60, degrees=True).as_matrix()
    x = torch.tensor(np.concatenate([R0[..., :, 0], R0[..., :, 1]], -1), dtype=dt, requires_grad=True)
    opt = torch.optim.Adam([x], lr=0.03)
    lim = []
    for it in range(iters):
        R = six_to_mat(x); G, P = fk(R)
        loss = 0.0
        for name, w, fn, d, tw in terms:
            mv = fn(G, P)
            if tw is not None:
                b = nz(tw(G, P)); mv = mv - (mv * b).sum(-1, keepdim=True) * b; d = d - (d * b).sum(-1, keepdim=True) * b
            e = 1 - (nz(mv) * nz(d)).sum(-1)                                     # 1 - cos
            loss = loss + w * (e / (1 + e / robust)).mean()                        # robust (bounded) penalty
        # priors: hinge knees / elbows, small toes, spine split evenly, neck/head split, no knee/elbow hyperextension
        rv = R.reshape(-1, 3, 3)
        def angvec(j):                                                            # small-angle vector of a local rotation
            Rj = R[:, j]; return 0.5 * torch.stack([Rj[:, 2, 1] - Rj[:, 1, 2], Rj[:, 0, 2] - Rj[:, 2, 0], Rj[:, 1, 0] - Rj[:, 0, 1]], -1)
        pr = 0.0
        for j in (4, 5):
            v = angvec(j); pr = pr + 0.5 * (v[:, 1] ** 2 + v[:, 2] ** 2).mean() + 2.0 * torch.relu(-v[:, 0]).pow(2).mean()
        vL, vR = angvec(18), angvec(19)
        pr = pr + 0.3 * (vL[:, 0] ** 2 + vL[:, 2] ** 2).mean() + 0.3 * (vR[:, 0] ** 2 + vR[:, 2] ** 2).mean()
        pr = pr + 2.0 * torch.relu(vL[:, 1]).pow(2).mean() + 2.0 * torch.relu(-vR[:, 1]).pow(2).mean()
        for j in (10, 11):
            pr = pr + 0.5 * (angvec(j) ** 2).sum(-1).mean()
        for j in (20, 21):
            pr = pr + 0.05 * (angvec(j) ** 2).sum(-1).mean()
        for j in (13, 14):
            pr = pr + 0.1 * (angvec(j) ** 2).sum(-1).mean()
        s1, s2, s3 = angvec(3), angvec(6), angvec(9)
        pr = pr + 0.5 * (((s1 - s2) ** 2).sum(-1) + ((s2 - s3) ** 2).sum(-1)).mean()
        pr = pr + 0.3 * ((angvec(12) - angvec(15)) ** 2).sum(-1).mean()
        sm = ((R[2:] - 2 * R[1:-1] + R[:-2]) ** 2).sum((-1, -2)).mean() if n > 2 else 0.0
        loss = loss + pr + w_smooth * J22 * sm
        opt.zero_grad(); loss.backward(); opt.step()
        if it == int(iters * 0.7):
            for g in opt.param_groups: g["lr"] = 0.01
        if log and it % 100 == 0:
            log(f"  it {it}: loss {float(loss.detach()):.5f} prior {float(pr.detach()):.5f} smooth {float(sm.detach()):.7f}")
    with torch.no_grad():
        R = six_to_mat(x); G, P = fk(R)
        res = {}
        for name, w, fn, d, tw in terms:
            mv = fn(G, P)
            if tw is not None:
                b = nz(tw(G, P)); mv = mv - (mv * b).sum(-1, keepdim=True) * b; d = d - (d * b).sum(-1, keepdim=True) * b
            ang = torch.rad2deg(torch.arccos((nz(mv) * nz(d)).sum(-1).clamp(-1, 1))).numpy()
            res[name] = {"median_deg": round(float(np.median(ang)), 2), "p95_deg": round(float(np.percentile(ang, 95)), 2),
                         "max_deg": round(float(ang.max()), 2)}
        return R.numpy(), G.numpy(), P.numpy(), res


def trial_landmarks(src, tid, sigma, kinect_cache="", tsv_dir=""):
    """one trial, or contiguous segments of one recording joined with '+' (the shared boundary sample is dropped)"""
    parts = tid.split("+")
    if src == "kinect":
        ts, Xs = [], []
        for k, p_ in enumerate(parts):
            t, X = load_kinect(kinect_cache, p_)
            if k:
                t = t[1:] - t[0] + ts[-1][-1] + (t[1] - t[0]); X = X[1:]
            ts.append(t); Xs.append(X)
        t, X = np.concatenate(ts), np.concatenate(Xs)
        tt, L, q0 = kinect_targets(t, X, sigma); dur = float(t[-1] - t[0])
    else:
        Xs = []
        for k, p_ in enumerate(parts):
            fs, names, X = load_qualisys(os.path.join(tsv_dir, p_ + ".npz"))
            Xs.append(X[1:] if k else X)
        X = np.concatenate(Xs)
        tt, L, q0 = qualisys_landmarks(fs, names, X, sigma); dur = (len(X) - 1) / fs
    q0["parts"] = parts
    return tt, L, q0, dur


def g01_trials(src, participant, kinect_cache="", tsv_dir=""):
    if src == "kinect":
        ids = [str(x) for x in np.load(kinect_cache)["trial_id"]]
    else:
        ids = [f[:-4] for f in os.listdir(tsv_dir) if f.endswith(".npz")]
    return sorted(t for t in ids if t[:3] == participant and t[10:12] == "01")


def cmd_fit(a):
    sk = load_skeleton(a.gender)
    tt, L, q0, dur = trial_landmarks(a.source, a.trial, a.sigma, a.kinect_cache, a.tsv_dir)
    g01 = g01_trials(a.source, a.trial[:3], a.kinect_cache, a.tsv_dir)
    if a.source == "kinect":                                       # same Kinect session where possible (the floor/tilt moved)
        same = [t for t in g01 if t[3:6] == a.trial[3:6]]; g01 = same or g01
    cal, per = calibrate_participant(a.source, g01, lambda t: trial_landmarks(a.source, t, a.sigma, a.kinect_cache, a.tsv_dir)[1])
    qual = quality(L, a.source)
    yaw = facing_yaw(L, a.source)
    R = yaw_rot(-yaw)
    pel = L["SpineBase"] if a.source == "kinect" else L["pelvis"]
    origin = np.array([pel[:, 0].mean(), 0.0, pel[:, 2].mean()])
    Lc = rotate_landmarks(L, R, origin)
    used = apply_calibration(Lc, a.source, cal, sk)
    print(f"{a.trial} {a.source}: {len(tt)} frames ({dur:.2f} s), facing {np.degrees(yaw):.1f} deg, quality {json.dumps(qual)}", flush=True)
    print(f"  calibration from {len(g01)} G01 trials: standing {json.dumps({k: round(v, 1) for k, v in cal.items()})}; rotation applied {used}")
    Rl, G, P, res = fit_trial(Lc, a.source, sk, iters=a.iters, w_smooth=a.w_smooth, log=print)
    out = {"R": Rl.astype(np.float32), "G": G.astype(np.float32), "P": P.astype(np.float32), "t": tt, "source": a.source, "trial": a.trial,
           "gender": a.gender, "duration_s": dur}
    keep = {k: v for k, v in Lc.items() if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 3 and len(v) == len(tt)}
    np.savez_compressed(a.out, **out, **{"L_" + k: v.astype(np.float32) for k, v in keep.items()},
                        info=json.dumps({"residual_deg": res, "quality": qual, "prep": q0, "facing_deg": float(np.degrees(yaw)),
                                         "origin": origin.tolist(), "sigma": a.sigma, "calibration": {"g01": g01, "standing": cal,
                                         "per_trial": per, "rotation_deg": used}}, default=float))
    print(json.dumps(res))


# ------------------------------------------------------------------------------------------------ contacts, root, leg IK
def _clean(mask, close=4, minlen=5):
    """fill gaps shorter than `close` frames, then drop runs shorter than `minlen`"""
    m = mask.copy(); n = len(m)
    for v, L_ in ((False, close), (True, minlen)):
        i = 0
        while i < n:
            if m[i] == v:
                j = i
                while j < n and m[j] == v: j += 1
                if j - i < L_ and (v or (i > 0 and j < n)): m[i:j] = not v
                i = j
            else:
                i += 1
    return m


def runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]: j += 1
            out.append((i, j - 1)); i = j
        else:
            i += 1
    return out


def data_contacts(L, src, h_thr=None, v_thr=None):
    """per foot: heel / ball contact flags from the data (Qualisys heel FCC + metatarsal markers; Kinect ankle + foot joints)"""
    out = {}
    for s_ in "LR":
        if src == "qualisys":
            pts = {"heel": L["heel" + s_], "ball": L["toe" + s_]}; ht, vt = h_thr or 0.015, v_thr or 0.15
            hts = {"heel": ht, "ball": 0.03}
        else:
            k = "Left" if s_ == "L" else "Right"
            pts = {"heel": L["Ankle" + k], "ball": L["Foot" + k]}; ht, vt = h_thr or 0.03, v_thr or 0.25
            hts = {"heel": ht, "ball": ht}
        c = {}
        for nm, X in pts.items():
            h = X[:, 1] - np.percentile(X[:, 1], 2)
            v = np.linalg.norm(np.gradient(X, axis=0), axis=1) * FPS
            c[nm] = _clean((h < hts[nm]) & (v < vt))
        if src == "kinect":                                    # Kinect cannot tell heel from toe: one flat contact
            f = _clean(c["heel"] | c["ball"]); c = {"heel": f.copy(), "ball": f.copy()}
        out[s_] = c
    return out


def two_bone(Rl, G, P, side, ankle_target, foot_G_target, J):
    """re-solve hip + knee (+ ankle) local rotations so the ankle reaches ankle_target and the foot gets the global
    orientation foot_G_target. Rl/G/P: local rotations, globals, positions [n,22,...] (modified copies returned)."""
    from scipy.spatial.transform import Rotation as Rot
    h, k, a = (1, 4, 7) if side == "L" else (2, 5, 8)
    Rl = Rl.copy()
    u = J[k] - J[h]; v = J[a] - J[k]
    d = np.linalg.norm(ankle_target - P[:, h], axis=1)
    dmax = (np.linalg.norm(u) + np.linalg.norm(v)) * 0.9995
    d = np.minimum(d, dmax)
    K0 = Rl[:, k]
    f = lambda th: np.linalg.norm(u + np.einsum("nab,nbc,c->na", K0, Rot.from_euler("x", th).as_matrix(), v), axis=1) - d
    phi = Rot.from_matrix(K0).as_rotvec()[:, 0]                 # current flexion (about the knee's x axis)
    lo = np.maximum(-np.radians(60), np.radians(3) - phi); hi = np.full(len(d), np.radians(140)) - np.maximum(phi, 0)
    for _ in range(50):                                          # monotone: total flexion stays >= 3 deg on [lo, hi]
        mid = (lo + hi) / 2; fm = f(mid)
        lo = np.where(fm > 0, mid, lo); hi = np.where(fm > 0, hi, mid)
    th = (lo + hi) / 2
    Rl[:, k] = K0 @ Rot.from_euler("x", th).as_matrix()
    # thigh: minimal rotation (in the world) taking the new hip->ankle vector to the target direction
    Gp = G[:, 0] if True else None
    Gh = G[:, h]
    cur = np.einsum("nab,b->na", Gh, u) + np.einsum("nab,nbc,c->na", Gh, Rl[:, k], v)
    tgt = ankle_target - P[:, h]
    ax_ = np.cross(cur, tgt); sn = np.linalg.norm(ax_, axis=1); cs = np.sum(cur * tgt, 1)
    ang = np.arctan2(sn, cs); ax_ = ax_ / np.maximum(sn, 1e-12)[:, None]
    Rw = Rot.from_rotvec(ax_ * ang[:, None]).as_matrix()
    Gh_new = Rw @ Gh
    Gpar = G[:, 0]                                               # hip parent = pelvis
    Rl[:, h] = np.swapaxes(Gpar, 1, 2) @ Gh_new
    Gk_new = Gh_new @ Rl[:, k]
    Rl[:, a] = np.swapaxes(Gk_new, 1, 2) @ foot_G_target
    return Rl


def fk_np(Rl, J, parents, root=None):
    n = len(Rl); G = np.zeros((n, J22, 3, 3)); P = np.zeros((n, J22, 3))
    for j in range(J22):
        p = parents[j]
        if p < 0:
            G[:, j] = Rl[:, j]; P[:, j] = 0 if root is None else root
        else:
            G[:, j] = G[:, p] @ Rl[:, j]; P[:, j] = P[:, p] + np.einsum("nab,b->na", G[:, p], J[j] - J[p])
    return G, P


def solve_root_and_feet(Rl, sk, root_track, contacts, w_track=1.0, w_contact=30.0, w_smooth=60.0, w_y=0.05, blend=6, log=print):
    """Root translation (least squares: follow the scaled data pelvis, keep contact points still and on the floor, smooth),
    then two-bone leg IK so every contact point sits exactly on its anchor (corrections faded in/out over `blend` frames).
    Floor = the model's rest sole height sk['sole_y']."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import lsqr
    from scipy.spatial.transform import Rotation as Rot
    J, par = sk["J"], sk["parents"]; n = len(Rl); floor = sk["sole_y"]
    G, P = fk_np(Rl, J, par)
    off = {}
    for s_ in "LR":
        a = 7 if s_ == "L" else 8; f_ = 10 if s_ == "L" else 11
        heel = sk["heel" + s_]; ball = np.array([J[f_][0], floor, J[f_][2]]) - J[a]
        off[s_] = {"heel": heel, "ball": ball, "a": a}
    pts = {s_: {nm: P[:, off[s_]["a"]] + np.einsum("nab,b->na", G[:, off[s_]["a"]], off[s_][nm]) for nm in ("heel", "ball")} for s_ in "LR"}
    # unknowns: T (n x 3), then one xz anchor per contact run per point
    anchors = []
    for s_ in "LR":
        for nm in ("heel", "ball"):
            for (i0, i1) in runs(contacts[s_][nm]):
                anchors.append((s_, nm, i0, i1))
    nv = 3 * n + 2 * len(anchors)
    rows, cols, vals, rhs = [], [], [], []
    def add(entries, b):
        r = len(rhs)
        for c_, v_ in entries: rows.append(r); cols.append(c_); vals.append(v_)
        rhs.append(b)
    for t in range(n):
        for d_ in (0, 2):
            add([(3 * t + d_, w_track)], w_track * root_track[t, d_])
        add([(3 * t + 1, w_y)], w_y * root_track[t, 1])
    for t in range(1, n - 1):
        for d_ in range(3):
            add([(3 * (t - 1) + d_, w_smooth), (3 * t + d_, -2 * w_smooth), (3 * (t + 1) + d_, w_smooth)], 0.0)
    for k_, (s_, nm, i0, i1) in enumerate(anchors):
        c0 = 3 * n + 2 * k_
        for t in range(i0, i1 + 1):
            q = pts[s_][nm][t]
            add([(3 * t + 0, w_contact), (c0, -w_contact)], -w_contact * q[0])
            add([(3 * t + 2, w_contact), (c0 + 1, -w_contact)], -w_contact * q[2])
            add([(3 * t + 1, w_contact)], w_contact * (floor - q[1]))
    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(rhs), nv))
    x = lsqr(A, np.array(rhs), atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    T = x[:3 * n].reshape(n, 3)
    anc = {(s_, nm, i0, i1): np.array([x[3 * n + 2 * k_], floor, x[3 * n + 2 * k_ + 1]]) for k_, (s_, nm, i0, i1) in enumerate(anchors)}
    # flat contacts: make every overlapping heel / ball anchor pair exactly one flat foot apart (the least-squares anchors
    # come from a foot whose pitch varies, so their horizontal separation is not the flat-foot length)
    for s_ in "LR":
        Lhb = np.linalg.norm((off[s_]["ball"] - off[s_]["heel"])[[0, 2]])
        for kh in [k for k in anc if k[0] == s_ and k[1] == "heel"]:
            for kb in [k for k in anc if k[0] == s_ and k[1] == "ball"]:
                if max(kh[2], kb[2]) <= min(kh[3], kb[3]):
                    ah, ab = anc[kh], anc[kb]; m_ = (ah + ab) / 2; u_ = ab - ah; u_[1] = 0; u_ /= max(np.linalg.norm(u_), 1e-9)
                    anc[kh] = m_ - u_ * Lhb / 2; anc[kb] = m_ + u_ * Lhb / 2
    rest_dir = {s_: np.arctan2(*(off[s_]["ball"] - off[s_]["heel"])[[0, 2]]) for s_ in "LR"}     # rest heel->ball yaw
    def leg_ik_pass(T):
        # leg IK: desired ankle position and foot orientation in every frame (identity correction outside contact, faded)
        Rout = Rl.copy()
        G, P = fk_np(Rout, J, par, T)
        info = {"anchors": len(anchors), "runs": {}}
        pivots.clear(); pivots.update({"L": {}, "R": {}})
        for s_ in "LR":
            a = off[s_]["a"]
            c_any = contacts[s_]["heel"] | contacts[s_]["ball"]
            dA = np.zeros((n, 3)); dRv = np.zeros((n, 3)); wgt = np.zeros(n)
            from scipy.ndimage import gaussian_filter1d as _gf
            for (i0, i1) in runs(c_any):
                # 1. raw target foot orientation per frame (flat: lying flat at the anchors' yaw; one point: as fitted)
                Graw, piv = [], []
                for t in range(i0, i1 + 1):
                    Gf = G[t, a]
                    he = contacts[s_]["heel"][t]; ba = contacts[s_]["ball"][t]
                    key = lambda nm: next(kk for kk in anc if kk[0] == s_ and kk[1] == nm and kk[2] <= t <= kk[3])
                    if he and ba:
                        ah, ab = anc[key("heel")], anc[key("ball")]
                        psi = np.arctan2(*(ab - ah)[[0, 2]]) - rest_dir[s_]
                        Graw.append(Rot.from_euler("y", psi).as_matrix()); piv.append(("heel", anc[key("heel")]))
                    else:
                        nm = "heel" if he else "ball"; Graw.append(Gf); piv.append((nm, anc[key(nm)]))
                # 2. smooth the orientation over the run (no snap when the contact switches between flat / heel / ball)
                Graw = np.stack(Graw); ref = Rot.from_matrix(Graw[0])
                rv = (ref.inv() * Rot.from_matrix(Graw)).as_rotvec()
                if len(rv) > 3:
                    rv = _gf(rv, 3.0, axis=0, mode="nearest")
                Gs = (ref * Rot.from_rotvec(rv)).as_matrix()
                # 3. the ankle follows from the anchored point and the smoothed orientation
                for k_, t in enumerate(range(i0, i1 + 1)):
                    nm, A_ = piv[k_]; At = A_ - Gs[k_] @ off[s_][nm]; pivots[s_][t] = (nm, A_)
                    dA[t] = At - P[t, a]; dRv[t] = Rot.from_matrix(Gs[k_] @ G[t, a].T).as_rotvec(); wgt[t] = 1
            # fade the corrections out over `blend` frames on both sides of every contact run
            for (i0, i1) in runs(c_any):
                for k_ in range(1, blend + 1):
                    w_ = 1 - minjerk_s(k_ / (blend + 1))
                    for t, e in ((i0 - k_, i0), (i1 + k_, i1)):
                        if 0 <= t < n and wgt[t] < w_:
                            dA[t] = w_ * dA[e]; dRv[t] = w_ * dRv[e]; wgt[t] = w_
            mv = wgt > 0
            if mv.any():
                At = P[:, a] + dA
                Gt = Rot.from_rotvec(dRv).as_matrix() @ G[:, a]
                Rnew = two_bone(Rout, G, P, s_, At, Gt, J)
                Rout[mv] = Rnew[mv]
                G, P = fk_np(Rout, J, par, T)
            info["runs"][s_] = runs(c_any)
            info["max_ik_shift_cm_" + s_] = round(100 * float(np.linalg.norm(dA, axis=1).max()), 2)
        return Rout, info, off
    T0 = T.copy(); pivots = {}
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d
    hist = []
    for it_ in range(5):
        Rout, info, off = leg_ik_pass(T)
        info['miss_per_pass_mm'] = hist
        G, P = fk_np(Rout, J, par, T)
        miss = np.zeros(n)
        for s_ in "LR":
            a = off[s_]["a"]
            for t, (nm, A_) in pivots[s_].items():                    # the point each frame is pinned by
                q = P[t, a] + G[t, a] @ off[s_][nm]
                miss[t] = max(miss[t], float(np.linalg.norm(q - A_)))
        info["root_lowered_mm"] = round(1e3 * float(np.max(T0[:, 1] - T[:, 1])), 2)
        info.setdefault("miss_per_pass_mm", []).append(round(1e3 * float(miss.max()), 2))
        if miss.max() < 1.5e-3 or it_ == 4:
            break
        drop = gaussian_filter1d(maximum_filter1d(miss, 9), 3) * 1.3          # a lower pelvis gives the legs the missing reach
        T = T.copy(); T[:, 1] -= drop
    # floor check on the contact points of the final pose
    G, P = fk_np(Rout, J, par, T)
    info["unreached_mm"] = round(1e3 * max([float(np.linalg.norm(P[t, off[s_]["a"]] + G[t, off[s_]["a"]] @ off[s_][nm] - A_))
                                            for s_ in "LR" for t, (nm, A_) in pivots[s_].items()] + [0.0]), 2)
    low = []
    for s_ in "LR":
        a = off[s_]["a"]
        for nm in ("heel", "ball"):
            low.append(P[:, a, 1] + np.einsum("nab,b->na", G[:, a], off[s_][nm])[:, 1])
    info["lowest_contact_point_above_floor_mm"] = round(1e3 * float(np.min(low) - floor), 2)
    if log: log(f"  root/feet: {len(anchors)} anchors, IK shift max L {info['max_ik_shift_cm_L']} cm R {info['max_ik_shift_cm_R']} cm, "
                f"lowest contact point {info['lowest_contact_point_above_floor_mm']} mm above the floor, worst anchor miss {info['unreached_mm']} mm, passes {info['miss_per_pass_mm']}, root lowered {info['root_lowered_mm']} mm")
    return Rout, T, info


def floor_pass(Rl, T, sk, contacts, log=None, iters=3):
    """No sole point below the floor in any frame.
    1. Toes: where the toe tip is below the floor, the toe joint (SMPL-X joint 10/11, at the ball) is dorsiflexed until the
       tip touches the floor (metatarsophalangeal extension, e.g. in a raised-heel 'empty step').
    2. Heel / ball: a foot touching with one point is pitched about that point until the other is on the floor; a foot in
       the air is lifted. Per-frame corrections are eased in/out (maximum filter + Gaussian) and applied by the two-bone IK."""
    from scipy.spatial.transform import Rotation as Rot
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d
    J, par = sk["J"], sk["parents"]; floor = sk["sole_y"]; n = len(Rl)
    R = Rl.copy(); info = {}
    for s_, a, f_ in (("L", 7, 10), ("R", 8, 11)):
        off_h = sk["heel" + s_]; off_b = np.array([J[f_][0], floor, J[f_][2]]) - J[a]
        off_t = sk["toe" + s_]                                                 # relative to the toe joint f_
        def pts(G, P):
            return (P[:, a] + np.einsum("nab,b->na", G[:, a], off_h), P[:, a] + np.einsum("nab,b->na", G[:, a], off_b),
                    P[:, f_] + np.einsum("nab,b->na", G[:, f_], off_t))
        G, P = fk_np(R, J, par, T)
        h0, b0, t0 = pts(G, P)
        before = float(floor - min(h0[:, 1].min(), b0[:, 1].min(), t0[:, 1].min()))
        for it in range(iters):
            # --- heel / ball
            G, P = fk_np(R, J, par, T); h, b, t = pts(G, P)
            dA = np.zeros((n, 3)); dR = np.zeros((n, 3)); need = np.zeros(n, bool)
            he = contacts[s_]["heel"]; ba = contacts[s_]["ball"]
            for k in range(n):
                ph, pb = floor - h[k, 1], floor - b[k, 1]
                if max(ph, pb) <= 3e-4:
                    continue
                need[k] = True; Ga, Pa = G[k, a], P[k, a]
                if he[k] != ba[k]:
                    piv_off, oth_off, sign = (off_h, off_b, -1.0) if he[k] else (off_b, off_h, 1.0)
                    pp = Pa + Ga @ piv_off; axis = Ga @ np.array([1.0, 0, 0])
                    lo, hi = 0.0, np.radians(45)
                    for _ in range(40):
                        mid = (lo + hi) / 2; G2 = Rot.from_rotvec(axis * sign * mid).as_matrix() @ Ga
                        P2 = pp - G2 @ piv_off
                        lo, hi = (mid, hi) if (P2 + G2 @ oth_off)[1] < floor else (lo, mid)
                    dR[k] = axis * sign * hi; G2 = Rot.from_rotvec(dR[k]).as_matrix() @ Ga; dA[k] = (pp - G2 @ piv_off) - Pa
                else:
                    dA[k, 1] = max(ph, pb)
            if need.any():
                env = gaussian_filter1d(maximum_filter1d(need.astype(float), 5), 1.5); env[need] = 1.0
                idx = np.where(need)[0]
                fill = lambda X: np.stack([np.interp(np.arange(n), idx, X[idx, c]) for c in range(3)], 1)
                dA_s = fill(dA) * env[:, None]; dR_s = fill(dR) * env[:, None]
                dA_s[need] = dA[need]; dR_s[need] = dR[need]
                At = P[:, a] + dA_s; Gt = np.einsum("nab,nbc->nac", Rot.from_rotvec(dR_s).as_matrix(), G[:, a])
                R = two_bone(R, G, P, s_, At, Gt, J)
            # --- toes
            G, P = fk_np(R, J, par, T); h, b, t = pts(G, P)
            low = t[:, 1] < floor - 3e-4
            if low.any():
                phi = np.zeros(n)
                for k in np.where(low)[0]:
                    lo, hi = 0.0, np.radians(40)                                     # LBS creases the sole beyond ~40 deg
                    for _ in range(40):
                        mid = (lo + hi) / 2
                        Gt_ = G[k, f_] @ Rot.from_euler("x", -mid).as_matrix()       # toe extension (tip up)
                        tip = P[k, f_] + Gt_ @ off_t
                        lo, hi = (mid, hi) if tip[1] < floor else (lo, mid)
                    phi[k] = hi
                env = gaussian_filter1d(maximum_filter1d(phi, 5), 1.5); env = np.maximum(env, phi)
                R[:, f_] = R[:, f_] @ Rot.from_euler("x", -env).as_matrix()
                # total toe extension capped at 40 deg (beyond it LBS folds the sole under the ball)
                rv_ = Rot.from_matrix(R[:, f_]).as_rotvec(); ang_ = np.linalg.norm(rv_, axis=1)
                over = ang_ > np.radians(40)
                if over.any():
                    rv_[over] *= (np.radians(40) / ang_[over])[:, None]; R[:, f_] = Rot.from_rotvec(rv_).as_matrix()
        G, P = fk_np(R, J, par, T); h, b, t = pts(G, P)
        after = float(floor - min(h[:, 1].min(), b[:, 1].min(), t[:, 1].min()))
        toe_ext = np.degrees(np.abs(Rot.from_matrix(R[:, f_]).as_rotvec()[:, 0]))
        info[s_] = {"max_penetration_mm_before": round(1e3 * max(before, 0), 2), "after": round(1e3 * max(after, 0), 2),
                    "toe_extension_max_deg": round(float(toe_ext.max()), 1)}
    if log: log(f"  floor pass: {info}")
    return R, info


def minjerk_s(s_):
    s_ = np.clip(s_, 0.0, 1.0)
    return s_ ** 3 * (10 - 15 * s_ + 6 * s_ * s_)


def mats_to_quats(R):
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(R.reshape(-1, 3, 3)).as_quat().reshape(R.shape[:-2] + (4,))   # (x, y, z, w)
    q *= np.where(q[..., 3:4] < 0, -1, 1)
    return q


def leg_length(L, src):
    seg = segment_lengths(L, src)
    return float(np.median((seg["thighL"] + seg["shankL"] + seg["thighR"] + seg["shankR"]) / 2))


def level_gaze(Rl, sk, target_deg=-2.0, quiet_pct=100):
    """Constant head pitch correction so that the gaze (head joint +Z) has elevation target_deg in the quiet frames (lowest
    quarter of wrist speed): the marker headband's tilt is calibrated on G01 standing, but from take to take it can sit
    differently. 30 % of the correction goes to the neck, 70 % to the head."""
    from scipy.spatial.transform import Rotation as Rot
    G, P = fk_np(Rl, sk["J"], sk["parents"])
    el = np.degrees(np.arcsin(np.clip((G[:, 15] @ np.array([0, 0, 1.0]))[:, 1], -1, 1)))
    v = np.linalg.norm(np.diff(P[:, [20, 21]], axis=0), axis=2).mean(1); q = np.concatenate([[False], v <= np.percentile(v, quiet_pct)])
    e = float(np.median(el[q])); d = np.radians(e - target_deg)
    R = Rl.copy()
    R[:, 12] = R[:, 12] @ Rot.from_euler("x", 0.3 * d).as_matrix(); R[:, 15] = R[:, 15] @ Rot.from_euler("x", 0.7 * d).as_matrix()
    G2, _ = fk_np(R, sk["J"], sk["parents"])
    el2 = np.degrees(np.arcsin(np.clip((G2[:, 15] @ np.array([0, 0, 1.0]))[:, 1], -1, 1)))
    info = {"quiet_elevation_before_deg": round(e, 2), "correction_deg": round(float(np.degrees(d)), 2),
            "quiet_elevation_after_deg": round(float(np.median(el2[q])), 2), "range_after_deg": [round(float(el2.min()), 1), round(float(el2.max()), 1)]}
    print(f"  gaze: {info}")
    return R, info


def cmd_motion(a):
    """fit npz -> contacts, root, leg IK -> motion npz (local quaternions, root translation, contact runs)"""
    z = np.load(a.fit, allow_pickle=False)
    src = str(z["source"]); gender = str(z["gender"]); sk = load_skeleton(gender)
    L = {k[2:]: z[k].astype(np.float64) for k in z.files if k.startswith("L_")}
    Rl = z["R"].astype(np.float64)
    if a.gaze is not None:
        Rl, gz = level_gaze(Rl, sk, a.gaze)
    s_ = sk["leg"] / leg_length(L, src)
    pel = L["SpineBase"] if src == "kinect" else L["pelvis"]
    track = s_ * pel
    con = data_contacts(L, src)
    # pass 1 without vertical tracking, pass 2 with the data's (scaled) vertical pelvis motion around pass 1's mean
    R1, T1, _ = solve_root_and_feet(Rl, sk, track, con, w_y=0.0, log=None)
    tr2 = track.copy(); tr2[:, 1] = track[:, 1] - track[:, 1].mean() + T1[:, 1].mean()
    if a.ik:                                                       # two-bone IK + floor pass, then (optionally) the polish
        R2, T2, info = solve_root_and_feet(Rl, sk, tr2, con, w_y=a.w_y)
        R2, info["floor_pass"] = floor_pass(R2, T2, sk, con, log=print)
        if a.polish_iters > 0:
            R2, T2, info["polish"] = polish(R2, T2, sk, con, circular=False, iters=a.polish_iters, log=print)
    else:                                                          # root from least squares, legs and feet by the torch polish
        _, T2, info = solve_root_and_feet(Rl, sk, tr2, con, w_y=a.w_y, log=None)
        T_ls = T2
        R2, T2, info["polish"] = polish(Rl, T_ls, sk, con, circular=False, iters=a.polish_iters, log=print,
                                        w_dsmooth=a.w_dsmooth, soft_sigma=a.soft_sigma)
    info["scale_model_over_data"] = round(s_, 4)
    if a.gaze is not None: info["gaze"] = gz
    info["contact_frames"] = {k: {nm: int(v.sum()) for nm, v in c.items()} for k, c in con.items()}
    print(json.dumps({k: v for k, v in info.items() if k != "runs"}))
    np.savez_compressed(a.out, Q=mats_to_quats(R2).astype(np.float64), root=T2, R=R2, fit_R=Rl, source=src, gender=gender,
                        trial=str(z["trial"]), contactL_heel=con["L"]["heel"], contactL_ball=con["L"]["ball"],
                        contactR_heel=con["R"]["heel"], contactR_ball=con["R"]["ball"], floor=sk["sole_y"],
                        info=json.dumps(info, default=float), fit_info=str(z["info"]))


GESTURES = {"G01": ("Beginning position (Wuji)", "无极起势"), "G02": ("Tree posture", "太极桩"), "G03": ("Open and close lotus flower", "开合莲花"),
            "G04": ("Bring sky and earth together", "天地合一"), "G05": ("Canalize energy", "引导能量"), "G06": ("Drive the monkey away", "倒卷肱"),
            "G07": ("Move hands like clouds", "云手"), "G08": ("Part the wild horse's mane", "野马分鬃"), "G09": ("Golden rooster stands on one leg", "金鸡独立"),
            "G10": ("Fair lady works shuttles", "玉女穿梭"), "G11": ("Kick with heel", "蹬脚"), "G12": ("Brush knee and twist step", "搂膝拗步"),
            "G13": ("Grasp the bird's tail", "揽雀尾")}
CITATION = ("Tits M, Laraba S, Caulier E, Tilmanne J, Dutoit T (2018). UMONS-TAICHI: a multimodal motion capture dataset of expertise "
            "in Taijiquan gestures. Data in Brief 19, 1214-1221. doi:10.1016/j.dib.2018.05.088; Zenodo record 2784581; CC BY-NC-SA 4.0")


def base_meta(src, gender):
    sensor = ("Qualisys optical motion capture (68 markers, 179 Hz)" if src == "qualisys" else "Kinect v2 skeleton (25 joints, 30 Hz)")
    return {"name": "Tai Chi sequence (from recordings)",
            "motion": "recorded (UMONS-TAICHI), fitted to SMPL-X; modified",
            "license_note": ("Body: SMPL-X (MPI-IS) under its research licence; decimated web mesh. Motion: derived from UMONS-TAICHI "
                             "(Tits et al. 2018), CC BY-NC-SA 4.0, modified (fitted to SMPL-X, smoothed, blended, feet cleaned)"),
            "source": f"UMONS-TAICHI {sensor}, medoid trials per form, fitted to SMPL-X {gender.lower()} (tools/umons_motion.py)",
            "citation": CITATION,
            "rootTranslationNote": "int16 millimetres; pelvis path from the recording (scaled to the model's leg length), solved so contact feet stay still on groundY",
            "motionNote": (f"derived from recordings of a person (UMONS-TAICHI {sensor}; medoid trials of skilled performers), "
                           "fitted to a parametric body (SMPL-X joint rotations from bone directions and segment axes), light smoothing, "
                           "feet cleaned with a contact detector and leg IK; not procedurally generated"),
            "loop": "cycle"}


def cmd_single(a):
    """one motion npz -> a one-segment sequence npz (for previews / the G07 milestone)"""
    z = np.load(a.motion, allow_pickle=False)
    Q = z["Q"]; F = len(Q); tid = str(z["trial"]); G = tid[10:13] if tid[10] == "G" else tid.split("+")[0][9:12]
    gid = tid.split("+")[0][9:12]
    en, zh = GESTURES[gid]
    meta = base_meta(str(z["source"]), str(z["gender"])); meta["loop"] = a.loop
    segs = [{"name": f"{en} ({zh})", "id": gid, "trial": tid, "start": 0, "end": F - 1}]
    np.savez_compressed(a.out, Q=Q, root=z["root"], gender=str(z["gender"]), segments=json.dumps(segs, ensure_ascii=False),
                        meta=json.dumps(meta, ensure_ascii=False), contactL=z["contactL_heel"] | z["contactL_ball"],
                        contactR=z["contactR_heel"] | z["contactR_ball"])
    print("wrote", a.out, F, "frames")


# ------------------------------------------------------------------------------------------------ sequence (L2)
LEG_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]


def rot_y(a):
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])


def yaw_of(v):
    return float(np.arctan2(v[0], v[2]))


def feet_state(Rl, T, sk, t):
    """world heel / ball points and foot rotation of both feet at frame t (trial frame)"""
    G, P = fk_np(Rl[t:t + 1], sk["J"], sk["parents"], T[t:t + 1])
    out = {}
    for s_, a, f_ in (("L", 7, 10), ("R", 8, 11)):
        ball = np.array([sk["J"][f_][0], sk["sole_y"], sk["J"][f_][2]]) - sk["J"][a]
        out[s_] = {"heel": P[0, a] + G[0, a] @ sk["heel" + s_], "ball": P[0, a] + G[0, a] @ ball, "G": G[0, a], "ankle": P[0, a],
                   "heel_off": sk["heel" + s_], "ball_off": ball}
    return out


def place(x, th, t):
    """world = Ry(th) x + t (x [...,3])"""
    return x @ rot_y(th).T + np.array([t[0], 0.0, t[1]])


def rigid2d(src, dst):
    """best rotation about +Y and xz translation mapping src points [k,3] onto dst [k,3] (least squares)"""
    a = src[:, [0, 2]]; b = dst[:, [0, 2]]
    ca, cb = a.mean(0), b.mean(0); A = a - ca; B = b - cb
    # angle th such that Ry(th) maps (x,z): x' = c x + s z, z' = -s x + c z
    num = np.sum(A[:, 1] * B[:, 0] - A[:, 0] * B[:, 1]); den = np.sum(A[:, 0] * B[:, 0] + A[:, 1] * B[:, 1])
    th = np.arctan2(num, den)
    R = rot_y(th)[[0, 2]][:, [0, 2]]
    t = cb - R @ ca
    return th, t


def facing_series(Rl):
    """pelvis yaw (rad) per frame from the pelvis global rotation (local joint 0)"""
    return np.arctan2(Rl[:, 0, 0, 2], Rl[:, 0, 2, 2])


def plan_transition(A_end, B_start, B_info, fixed=None, step_cost=1.5, onA=None, onB=None):
    """choose 0 / 1 / 2 steps and B's placement (th, t). A_end: world feet of A's last frame; B_start: B's first-frame
    feet in B's own frame; B_info: mean facing (rad) and mean pelvis xz of B in its own frame. fixed: (th, t) for the seam."""
    pts = lambda F, sides=("L", "R"): np.stack([F[s_][k] for s_ in sides for k in ("heel", "ball")])
    opts = []
    def cost(th, t, n):
        face = np.degrees(np.angle(np.exp(1j * (B_info["facing"] + th))))
        pel = place(np.array([B_info["pelvis"][0], 0, B_info["pelvis"][1]]), th, t)[[0, 2]]
        feet = place(B_info["feet"], th, t)[:, [0, 2]]; rad = float(np.linalg.norm(feet, axis=1).max())
        # the hero camera sits 30 deg toward the subject's left: facing anywhere in about -15..+45 deg reads well
        return (((face - 15.0) / 30.0) ** 2 + float(pel @ pel) / 0.25 ** 2 + 30.0 * max(0.0, rad - 0.5) ** 2 / 0.05 ** 2
                + step_cost * n), face, pel
    cands = []
    if fixed is None:
        th, t = rigid2d(pts(B_start), pts(A_end)); cands.append(("none", th, t))
        for s_ in "LR":
            th, t = rigid2d(pts(B_start, (s_,)), pts(A_end, (s_,))); cands.append((s_, th, t))
        cands.append(("both", -B_info["facing"], None))
    else:
        cands.append(("fixed", fixed[0], fixed[1]))
    for kind, th, t in cands:
        if kind == "both":
            c_ = np.array(B_info["pelvis"]); R2 = rot_y(th)[[0, 2]][:, [0, 2]]; t = -R2 @ c_
        Bw = {s_: {k: place(B_start[s_][k], th, t) for k in ("heel", "ball")} for s_ in "LR"}
        miss = {s_: max(np.linalg.norm((Bw[s_][k] - A_end[s_][k])[[0, 2]]) for k in ("heel", "ball")) for s_ in "LR"}
        moving = [s_ for s_ in "LR" if miss[s_] > 0.015 or not (onA or {"L": 1, "R": 1})[s_] or not (onB or {"L": 1, "R": 1})[s_]]
        # feasibility: the two feet never closer than 12 cm (centre to centre) at the start, end and mid-step
        c = lambda F, s_: (F[s_]["heel"] + F[s_]["ball"]) / 2
        ok = all(np.linalg.norm((c(F_, "L") - c(F_, "R"))[[0, 2]]) >= 0.08 for F_ in (A_end, Bw))
        J_, face, pel = cost(th, t, len(moving))
        opts.append({"kind": kind, "th": float(th), "t": [float(t[0]), float(t[1])], "moving": moving, "miss_cm": {k: round(100 * v, 1) for k, v in miss.items()},
                     "cost": float(J_) + (0 if ok else 50), "facing_deg": round(float(face), 1), "pelvis_cm": [round(100 * float(x), 1) for x in pel], "feasible": ok})
    best = min(opts, key=lambda o: o["cost"])
    if len(best["moving"]) == 2:
        # step order: a foot in the air at A's end goes first, a foot in the air at B's start goes last; otherwise the foot
        # that has farther to go steps second
        key = lambda s_: (0 if onA and not onA[s_] else 2 if onB and not onB[s_] else 1, best["miss_cm"][s_])
        best["moving"] = sorted(best["moving"], key=key)
    return best, opts


def blend_rot(Ra_last, Ra_prev, Rb_first, Rb_next, W):
    """W transition rotations between A's last and B's first frame (exclusive), C1: each end pose is extrapolated with its
    own angular velocity (saturating), and the two extrapolations are slerped with minimum-jerk weights."""
    from scipy.spatial.transform import Rotation as Rot, Slerp
    wa = Rot.from_matrix(Ra_prev.T @ Ra_last).as_rotvec(); wb = Rot.from_matrix(Rb_first.T @ Rb_next).as_rotvec()
    c = W / 4.0; out = []
    for k in range(1, W + 1):
        u = k / (W + 1)
        ta = c * (1 - np.exp(-k / c)); tb = c * (1 - np.exp(-(W + 1 - k) / c))
        A = Ra_last @ Rot.from_rotvec(wa * ta).as_matrix(); B = Rb_first @ Rot.from_rotvec(-wb * tb).as_matrix()
        r = Slerp([0, 1], Rot.from_matrix(np.stack([A, B])))([minjerk_s(u)]).as_matrix()[0]
        out.append(r)
    return np.stack(out)


def hermite(p0, v0, p1, v1, W):
    out = []
    for k in range(1, W + 1):
        u = k / (W + 1); L_ = W + 1
        h00 = 2 * u ** 3 - 3 * u ** 2 + 1; h10 = u ** 3 - 2 * u ** 2 + u; h01 = -2 * u ** 3 + 3 * u ** 2; h11 = u ** 3 - u ** 2
        out.append(h00 * p0 + h10 * L_ * v0 + h01 * p1 + h11 * L_ * v1)
    return np.stack(out)


def _phase(u, a, b):
    return minjerk_s((u - a) / max(b - a, 1e-9))


def make_transition(A, B, plan, sk, W_step=54, W_none=36, lift=0.035, shift=0.6, pitch_deg=12.0, onA=None, onB=None,
                    min_sep=0.19):
    """Synthesise the transition from A's last frame to B's first frame (both placed in the world), Tai Chi cadence.
    Per step (W_step frames): weight onto the support foot first (pelvis over it, quintic, u 0..0.28), then the foot lifts
    (u 0.30), travels low (heel <= `lift` above the floor; path bulged so the feet stay >= min_sep apart), lands heel-first
    with the toes up (u 0.72), rolls flat (u 0.84), and the weight follows (u 0.78..1). Upper body: velocity-matched
    blend over the whole window. Legs: two-bone IK to the foot targets. Returns R, T, contact flags per point, notes."""
    from scipy.spatial.transform import Rotation as Rot, Slerp
    J, par = sk["J"], sk["parents"]
    moving = plan["moving"]; W = W_none if not moving else W_step * len(moving)
    Rt = np.zeros((W, J22, 3, 3))
    for j in range(J22):
        Rt[:, j] = blend_rot(A["R"][-1, j], A["R"][-2, j], B["R"][0, j], B["R"][1, j], W)
    vA = A["T"][-1] - A["T"][-2]; vB = B["T"][1] - B["T"][0]
    Tt = hermite(A["T"][-1], vA, B["T"][0], vB, W)
    FA = feet_state(A["R"], A["T"], sk, len(A["R"]) - 1); FB = feet_state(B["R"], B["T"], sk, 0)
    con = {s_ + p_: np.ones(W, bool) for s_ in "LR" for p_ in "hb"}
    tgt = {s_: [] for s_ in "LR"}
    for s_ in "LR":
        sl0 = Slerp([0, 1], Rot.from_matrix(np.stack([FA[s_]["G"], FB[s_]["G"]])))
        for k in range(W):
            if s_ in moving:
                tgt[s_].append((FA[s_]["heel"], FA[s_]["G"])); continue
            w = minjerk_s((k + 1) / (W + 1))
            tgt[s_].append(((1 - w) * FA[s_]["heel"] + w * FB[s_]["heel"], sl0([w]).as_matrix()[0]))
    centre = lambda F_, s_: (F_[s_]["heel"] + F_[s_]["ball"]) / 2
    for i, s_ in enumerate(moving):
        k0 = i * W_step; o = "R" if s_ == "L" else "L"
        airA = bool(onA is not None and not onA[s_]); airB = bool(onB is not None and not onB[s_])
        uL, uS, uF = (0.0 if airA else 0.30), 0.72, 0.84             # lift-off, heel strike, foot flat
        sup = centre(FA, o) if i == 0 else centre(FB, o)
        sl = Slerp([0, 1], Rot.from_matrix(np.stack([FA[s_]["G"], FB[s_]["G"]])))
        for k in range(k0, W):
            u = (k - k0 + 1) / (W_step + 1)
            if k >= k0 + W_step:                                     # after this step's window: at the landing pose
                u = 1.0
            if u < uL:
                continue
            sw = min(1.0, (u - uL) / (uS - uL)); w = minjerk_s(sw)
            heel = (1 - w) * FA[s_]["heel"] + w * FB[s_]["heel"]
            if not (airA or airB):
                heel = heel + np.array([0, lift * np.sin(np.pi * sw) ** 2, 0])
            cen = heel + 0.5 * ((1 - w) * (FA[s_]["ball"] - FA[s_]["heel"]) + w * (FB[s_]["ball"] - FB[s_]["heel"]))
            dv = cen - sup; dv[1] = 0; dd = np.linalg.norm(dv)
            if dd < min_sep and sw < 1.0:
                heel = heel + (dv / max(dd, 1e-6)) * (min_sep - dd) * np.sin(np.pi * sw) ** 2
            Gk = sl([w]).as_matrix()[0]
            th = 0.0                                                 # toes up for a heel-first landing, then roll flat
            if not airB:
                if 0.51 <= u < uS: th = pitch_deg * _phase(u, 0.51, uS)
                elif uS <= u < uF: th = pitch_deg * (1 - _phase(u, uS, uF))
            if th:
                Gk = Rot.from_rotvec(Gk @ np.array([1.0, 0, 0]) * -np.radians(th)).as_matrix() @ Gk
            tgt[s_][k] = (heel, Gk)
            if airB or u < uS:
                con[s_ + "h"][k] = False; con[s_ + "b"][k] = False
            elif u < uF:
                con[s_ + "b"][k] = False
        if airA:
            con[s_ + "h"][:k0] = False; con[s_ + "b"][:k0] = False
        # weight transfer: onto the support foot before the lift, held during the swing, then on to the next stance. The
        # amplitude is the largest (<= shift) that keeps both legs within 95 % of their length (no stretched-straight knee)
        Lleg = np.linalg.norm(J[4] - J[1]) + np.linalg.norm(J[7] - J[4])
        hipj = {"L": 1, "R": 2}; ank = {"L": 7, "R": 8}
        ks = np.arange(k0, k0 + W_step); us = (ks - k0 + 1) / (W_step + 1)
        pl = np.array([_phase(u, 0.0, 0.28) if u < 0.28 else 1.0 if u < 0.78 else 1 - _phase(u, 0.78, 1.0) for u in us])
        d = sup[None] - Tt[ks]; d[:, 1] = 0
        for amp in (shift, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0):
            Tc = Tt[ks] + amp * pl[:, None] * d
            ok = True
            for leg in "LR":
                hp = Tc + np.einsum("nab,b->na", Rt[ks, 0], J[hipj[leg]] - J[0])
                at = np.stack([tgt[leg][k][0] for k in ks]) - np.einsum("nab,b->na", np.stack([tgt[leg][k][1] for k in ks]), sk["heel" + leg])
                reach = np.linalg.norm(at - hp, axis=1) / Lleg
                if (reach > 0.95).any(): ok = False
            if ok or amp == 0.0:
                Tt[ks] = Tc; break
    G, P = fk_np(Rt, J, par, Tt)
    for s_ in "LR":
        heel_off = sk["heel" + s_]
        Gt = np.stack([g for _, g in tgt[s_]]); At = np.stack([h for h, _ in tgt[s_]]) - np.einsum("nab,b->na", Gt, heel_off)
        Rt = two_bone(Rt, G, P, s_, At, Gt, J); G, P = fk_np(Rt, J, par, Tt)
    # light low-pass of the synthesised leg rotations (sigma 2 frames, ends pinned to A's last / B's first frame); the
    # sequence polish then re-pins the feet
    from scipy.ndimage import gaussian_filter1d
    for j in LEG_JOINTS:
        Rj = Rot.from_matrix(np.concatenate([A["R"][-1:, j], Rt[:, j], B["R"][:1, j]])); M_ = Rj.mean()
        v = (M_.inv() * Rj).as_rotvec(); pad_ = 6
        vp = np.concatenate([np.repeat(v[:1], pad_, 0), v, np.repeat(v[-1:], pad_, 0)])
        vs = gaussian_filter1d(vp, 2.0, axis=0)[pad_:-pad_]
        wgt = np.sin(np.linspace(0, np.pi, len(v))) ** 0.5                 # 0 at both pinned ends
        v2 = v * (1 - wgt[:, None]) + vs * wgt[:, None]
        Rt[:, j] = (M_ * Rot.from_rotvec(v2[1:-1])).as_matrix()
    return Rt, Tt, con, {"frames": W, "moving": moving}


def cmd_sequence(a):
    """trials (motion npz, play order) -> one closed loop with planned transition steps -> sequence npz for build3"""
    files = a.motions.split(",")
    trials = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        trials.append({"R": z["R"].astype(np.float64), "T": z["root"].astype(np.float64), "trial": str(z["trial"]), "gender": str(z["gender"]),
                       "source": str(z["source"]), "cL": z["contactL_heel"] | z["contactL_ball"], "cR": z["contactR_heel"] | z["contactR_ball"],
                       "cLh": z["contactL_heel"], "cLb": z["contactL_ball"], "cRh": z["contactR_heel"], "cRb": z["contactR_ball"]})
    gender = trials[0]["gender"]; sk = load_skeleton(gender)
    for tr in trials:
        # trim the quiet head / tail (the performer waiting between techniques): mean joint speed below a quarter of the
        # trial's median, at most a.trim_max seconds per end
        G, P = fk_np(tr["R"], sk["J"], sk["parents"], tr["T"])
        v = np.linalg.norm(np.diff(P, axis=0), axis=2).mean(1); med = np.median(v); mx = int(a.trim_max * FPS)
        i0 = 0
        while i0 < mx and v[i0] < 0.25 * med: i0 += 1
        i1 = len(v)
        while len(v) - i1 < mx and v[i1 - 1] < 0.25 * med: i1 -= 1
        tr["trim"] = (i0, len(tr["R"]) - 1 - i1)
        for k in ("R", "T", "cL", "cR", "cLh", "cLb", "cRh", "cRb"):
            tr[k] = tr[k][i0:i1 + 1]
        fy = facing_series(tr["R"]); tr["facing"] = float(np.angle(np.mean(np.exp(1j * fy))))
        tr["pelvis"] = [float(tr["T"][:, 0].mean()), float(tr["T"][:, 2].mean())]
        G, P = fk_np(tr["R"], sk["J"], sk["parents"], tr["T"])
        tr["feet"] = np.concatenate([P[::5, 7], P[::5, 8], P[::5, 10], P[::5, 11]])
    # placements
    N = len(trials); pl = [(0.0, np.zeros(2))]; plans = []
    trials[0]["th"], trials[0]["t"] = 0.0, np.zeros(2)
    def placed(tr):
        Ry = rot_y(tr["th"]); R = tr["R"].copy(); R[:, 0] = np.einsum("ab,nbc->nac", Ry, R[:, 0])
        return {"R": R, "T": place(tr["T"], tr["th"], tr["t"])}
    for i in range(1, N + 1):
        A = placed(trials[i - 1]); B = trials[i % N]
        FA = feet_state(A["R"], A["T"], sk, len(A["R"]) - 1); FB = feet_state(B["R"], B["T"], sk, 0)
        trA = trials[i - 1]
        onA = {"L": bool(trA["cL"][-1]), "R": bool(trA["cR"][-1])}; onB = {"L": bool(B["cL"][0]), "R": bool(B["cR"][0])}
        best, opts = plan_transition(FA, FB, {"facing": B["facing"], "pelvis": B["pelvis"], "feet": B["feet"]},
                                     fixed=(0.0, np.zeros(2)) if i == N else None, onA=onA, onB=onB)
        best["onA"], best["onB"] = onA, onB
        if i < N:
            B["th"], B["t"] = best["th"], np.array(best["t"])
        plans.append({"from": trials[i - 1]["trial"], "to": B["trial"], "chosen": best, "options": opts})
        print(f"transition {trials[i - 1]['trial'][9:12]} -> {B['trial'][9:12]}: {best['kind']}, steps {best['moving']}, "
              f"facing {best['facing_deg']} deg, pelvis {best['pelvis_cm']} cm, miss {best['miss_cm']}")
    # assemble: [seam transition, trial 0, transition, trial 1, ...]
    P_ = [placed(tr) for tr in trials]
    parts_R, parts_T, cL, cR, segs = [], [], [], [], []
    cp = {k: [] for k in ("cLh", "cLb", "cRh", "cRb")}
    f = 0
    for i in range(N):
        A = P_[i - 1]; B = P_[i]
        ch = plans[i - 1]["chosen"]
        Rt, Tt, con, note = make_transition(A, B, ch, sk, W_step=a.step_frames, W_none=a.blend_frames, onA=ch["onA"], onB=ch["onB"])
        tid = trials[i]["trial"]; gid = tid.split("+")[0][9:12]; en, zh = GESTURES[gid]
        n_tr = len(Rt); n = len(B["R"])
        segs.append({"name": f"{en} ({zh})", "id": gid, "trial": tid, "start": f, "end": f + n_tr + n - 1,
                     "transition_frames": n_tr, "transition_steps": note["moving"], "trial_frames": n,
                     "trimmed_frames_head_tail": list(trials[i]["trim"])})
        parts_R += [Rt, B["R"]]; parts_T += [Tt, B["T"]]
        cL += [con["Lh"] | con["Lb"], trials[i]["cL"]]; cR += [con["Rh"] | con["Rb"], trials[i]["cR"]]
        for k in cp: cp[k] += [con[k[1] + k[2]], trials[i][k]]
        segs[-1]["trial_start"] = f + n_tr; segs[-1]["trial_end"] = f + n_tr + n - 1
        f += n_tr + n
    R = np.concatenate(parts_R); T = np.concatenate(parts_T)
    cLa, cRa = np.concatenate(cL), np.concatenate(cR)
    cpa = {k: np.concatenate(v) for k, v in cp.items()}
    # loop start: frame 0 = the opening form's first recorded frame with both feet flat and the hands low (what a visitor
    # sees first); the seam transition then closes the last segment
    s0 = segs[0]; Fn = len(R); seam = s0["transition_frames"]; ts, te = s0["trial_start"], s0["trial_end"]
    G_, P_ = fk_np(R, sk["J"], sk["parents"], T)
    hands_all = (P_[:, 20, 1] + P_[:, 21, 1]) / 2 - sk["sole_y"]
    flat = cpa["cLh"] & cpa["cLb"] & cpa["cRh"] & cpa["cRb"]
    cand = [f for f in range(ts, te + 1) if flat[f]]
    f0 = int(min(cand, key=lambda f: hands_all[f])) if cand else ts
    k0 = f0 - ts                                                  # recorded frames of the opening form before the loop start
    roll = lambda X: np.roll(X, -f0, axis=0)
    R, T, cLa, cRa = roll(R), roll(T), roll(cLa), roll(cRa); cpa = {k: roll(v) for k, v in cpa.items()}
    for sg in segs:
        for k in ("start", "end", "trial_start", "trial_end"):
            sg[k] = (sg[k] - f0) % Fn
    head = dict(segs[0])
    segs[0].update({"start": 0, "trial_start": 0, "transition_frames": 0, "trial_phase_start_frame": k0})
    if k0 > 0:                                                    # the opening form's first frames close the loop
        head.update({"start": Fn - k0, "end": Fn - 1, "trial_start": Fn - k0, "trial_end": Fn - 1, "transition_frames": 0,
                     "trial_phase_start_frame": 0, "continues_at_frame_0": True})
        segs[-1]["end"] = Fn - k0 - 1; segs[-1]["seam_transition_frames"] = seam
        segs.append(head)
    else:
        segs[-1]["end"] = Fn - 1; segs[-1]["seam_transition_frames"] = seam
    hands = float(hands_all[f0]); flat0 = bool(flat[f0])
    print(f"loop start: assembled frame {f0} = recorded frame {k0} of {s0['id']} (lowest hands with both feet flat): wrists {hands:.2f} m above the floor")
    if a.floor_pass:
        R, fp = floor_pass(R, T, sk, {"L": {"heel": cpa["cLh"], "ball": cpa["cLb"]}, "R": {"heel": cpa["cRh"], "ball": cpa["cRb"]}}, log=print)
    meta = base_meta(trials[0]["source"], gender)
    meta["segmentsNote"] = ("start/end are inclusive frame indices; frame 0 is the opening form's first recorded frame; every later "
                            "segment opens with a synthesised transition (blend of the end and start poses plus 0-2 planned steps at "
                            "a Tai Chi cadence), then plays one recorded trial at its real speed; the last segment ends with the "
                            "transition back to the opening form")
    np.savez_compressed(a.out, Q=mats_to_quats(R), root=T, gender=gender, segments=json.dumps(segs, ensure_ascii=False),
                        meta=json.dumps(meta, ensure_ascii=False), contactL=cLa, contactR=cRa,
                        contactL_heel=cpa["cLh"], contactL_ball=cpa["cLb"], contactR_heel=cpa["cRh"], contactR_ball=cpa["cRb"],
                        plans=json.dumps(plans, default=float))
    print(f"wrote {a.out}: {len(R)} frames = {len(R) / FPS:.1f} s; segments " + ", ".join(f"{s_['id']} {s_['start']}-{s_['end']}" for s_ in segs))


def sole_points(sk):
    """per foot: (joint, rest offset) of heel, inner ball, outer ball, toe tip; all on the rest sole plane"""
    J = sk["J"]; out = {}
    M = np.load(os.path.join(MODELS, f"SMPLX_{sk['gender']}.npz"), allow_pickle=True)
    vt = M["v_template"].astype(np.float64); vt -= (M["J_regressor"].astype(np.float64) @ vt)[0]; lab = M["weights"].argmax(1)
    for s_, a, f_ in (("L", 7, 10), ("R", 8, 11)):
        v = vt[np.isin(lab, [a, f_])]; y0 = v[:, 1].min()
        nb = v[np.abs(v[:, 2] - J[f_][2]) < 0.025]
        inner = nb[np.argmin(np.abs(nb[:, 0]))] if s_ == "L" else nb[np.argmin(np.abs(nb[:, 0]))]
        outer = nb[np.argmax(np.abs(nb[:, 0]))]
        pts = {"heel": (a, np.array([sk["heel" + s_][0], y0 - J[a][1], sk["heel" + s_][2]])),
               "ball_in": (a, np.array([inner[0], y0, inner[2]]) - J[a]), "ball_out": (a, np.array([outer[0], y0, outer[2]]) - J[a]),
               "toe": (f_, np.array([0, 0, 0]) + sk["toe" + s_] * np.array([1, 0, 1]) + np.array([0, y0 - J[f_][1], 0]))}
        out[s_] = pts
    return out


def polish(R0n, T0n, sk, flags, circular=True, iters=1000, lr=0.002, w_pose=1.0, w_leg=0.15, w_smooth=30.0, toe_max=40.0, log=print,
           w_dsmooth=3e3, soft_sigma=2.0):
    """torch cleanup: stay close to (R0n, T0n), keep flagged contact points still and on the floor, every sole point above
    the floor, no knee hyperextension, toe extension <= toe_max, smooth (circular if the motion loops).
    R0n [n,22,3,3], T0n [n,3], flags {side: {"heel": bool[n], "ball": bool[n]}}."""
    import torch
    torch.set_num_threads(4); dt = torch.float64
    floor = sk["sole_y"]; n = len(R0n)
    R0 = torch.tensor(R0n, dtype=dt); T0 = torch.tensor(T0n, dtype=dt)
    J = torch.tensor(sk["J"], dtype=dt); par = sk["parents"]
    pts = sole_points(sk)
    fl = {s_: {k: torch.tensor(np.asarray(v, bool)) for k, v in flags[s_].items()} for s_ in "LR"}
    # soft contact weights: a contact fades in / out over a few frames instead of switching on (no jerk at heel strike)
    from scipy.ndimage import gaussian_filter1d as _g
    soft = {s_: {k: torch.tensor(np.asarray(v, float) * _g(np.asarray(v, float), soft_sigma, mode="wrap" if circular else "nearest"))
                 for k, v in flags[s_].items()} for s_ in "LR"}
    dlt = torch.zeros(n, J22, 3, dtype=dt, requires_grad=True); tau = torch.zeros(n, 3, dtype=dt, requires_grad=True)
    def expmap(v):
        th = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-12); k = v / th
        K = torch.zeros(v.shape[:-1] + (3, 3), dtype=dt)
        K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
        K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
        th = th[..., None]
        return torch.eye(3, dtype=dt) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)
    def fk(R, T):
        G = [None] * J22; P = [None] * J22
        for j in range(J22):
            p = par[j]
            if p < 0: G[j] = R[:, j]; P[j] = T
            else: G[j] = G[p] @ R[:, j]; P[j] = P[p] + G[p] @ (J[j] - J[p])
        return torch.stack(G, 1), torch.stack(P, 1)
    wj = torch.ones(J22, dtype=dt)
    for j in LEG_JOINTS: wj[j] = w_leg
    opt = torch.optim.Adam([dlt, tau], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.05)
    pad = (lambda X: torch.cat([X[-2:], X, X[:2]])) if circular else (lambda X: X)
    offs = {s_: {nm: (j, torch.tensor(off, dtype=dt)) for nm, (j, off) in pts[s_].items()} for s_ in "LR"}
    cos_toe = float(np.cos(np.radians(toe_max) / 2))
    f_ = lambda x: float(x.detach()) if hasattr(x, "detach") else float(x)
    for it in range(iters):
        R = R0 @ expmap(dlt); T = T0 + tau
        G, P = fk(R, T)
        l_pose = (wj[None, :, None] * dlt ** 2).sum(-1).mean() * J22
        l_root = (tau ** 2).sum(-1).mean() * 100.0
        l_c = 0.0; l_f = 0.0; l_cy = 0.0
        for s_ in "LR":
            for nm, (j, off) in offs[s_].items():
                w = G[:, j] @ off + P[:, j]
                l_f = l_f + torch.relu(floor - w[:, 1]).pow(2).mean()
                if nm == "toe":
                    continue
                kk = "heel" if nm == "heel" else "ball"; m_ = fl[s_][kk]; sw_ = soft[s_][kk]
                v = w[1:] - w[:-1]; m = (m_[1:] & m_[:-1])
                if m.any():
                    ws = torch.minimum(sw_[1:], sw_[:-1])
                    l_c = l_c + ((v ** 2).sum(-1) * ws).sum() / ws.sum().clamp_min(1)
                    l_cy = l_cy + (((w[:, 1] - floor) ** 2) * sw_).sum() / sw_.sum().clamp_min(1)
        kx = lambda Rj: 0.5 * (Rj[:, 2, 1] - Rj[:, 1, 2])
        l_k = torch.relu(np.sin(np.radians(3)) - kx(R[:, 4])).pow(2).mean() + torch.relu(np.sin(np.radians(3)) - kx(R[:, 5])).pow(2).mean()
        # toe rotation angle: cos(theta/2) = sqrt(1 + tr)/2 ; penalise beyond toe_max
        for j in (10, 11):
            tr = R[:, j, 0, 0] + R[:, j, 1, 1] + R[:, j, 2, 2]
            l_k = l_k + torch.relu(cos_toe - torch.sqrt(torch.clamp(1 + tr, min=1e-9)) / 2).pow(2).mean() * 10
        Rc = pad(R); Tc = pad(T)
        l_s = ((Rc[2:] - 2 * Rc[1:-1] + Rc[:-2]) ** 2).sum((-1, -2)).mean() * J22 + ((Tc[2:] - 2 * Tc[1:-1] + Tc[:-2]) ** 2).sum(-1).mean() * 1e3
        dc = pad(dlt); tc = pad(tau)                               # the corrections themselves must be smooth (low-frequency)
        l_ds = ((dc[2:] - 2 * dc[1:-1] + dc[:-2]) ** 2).sum(-1).mean() * J22 + ((tc[2:] - 2 * tc[1:-1] + tc[:-2]) ** 2).sum(-1).mean() * 100
        loss = w_pose * l_pose + l_root + 1e5 * l_c + 1e4 * l_cy + 1e6 * l_f + 1e3 * l_k + w_smooth * l_s + w_dsmooth * l_ds
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log and (it % 250 == 0 or it == iters - 1):
            log(f"  polish it {it}: pose {f_(l_pose):.5f} root {f_(l_root):.5f} contact-vel {f_(l_c):.2e} contact-y {f_(l_cy):.2e} "
                f"floor {f_(l_f):.2e} knee/toe {f_(l_k):.2e} smooth {f_(l_s):.2e}")
    with torch.no_grad():
        R = (R0 @ expmap(dlt)).numpy(); T = (T0 + tau).numpy(); ang = np.degrees(torch.linalg.norm(dlt, dim=-1).numpy())
    info = {"max_rotation_change_deg": round(float(ang.max()), 2), "p99_rotation_change_deg": round(float(np.percentile(ang, 99)), 2),
            "max_root_change_mm": round(1e3 * float(np.abs(T - T0n).max()), 2)}
    if log: log(f"  polish: {info}")
    return R, T, info


def cmd_polish(a):
    z = np.load(a.seq, allow_pickle=False)
    gender = str(z["gender"]); sk = load_skeleton(gender)
    from scipy.spatial.transform import Rotation as Rot
    Q = z["Q"].astype(np.float64); n = len(Q)
    R0 = Rot.from_quat(Q.reshape(-1, 4)).as_matrix().reshape(n, J22, 3, 3)
    flags = {s_: {"heel": z[f"contact{s_}_heel"], "ball": z[f"contact{s_}_ball"]} for s_ in "LR"}
    R, T, info = polish(R0, z["root"], sk, flags, circular=True, iters=a.iters, lr=a.lr, w_pose=a.w_pose, w_leg=a.w_leg, w_smooth=a.w_smooth,
                        w_dsmooth=a.w_dsmooth, soft_sigma=a.soft_sigma)
    out = {k: z[k] for k in z.files}
    out["Q"] = mats_to_quats(R); out["root"] = T; out["polish"] = json.dumps(info)
    np.savez_compressed(a.out, **out)


# ------------------------------------------------------------------------------------------------ card B bands
STATURE_CM = {"P01": 196, "P02": 163, "P03": 162, "P04": 150, "P05": 163, "P06": 180, "P07": 157, "P08": 158, "P09": 178,
              "P10": 163, "P11": 165, "P12": 181}            # UMONS-TAICHI Metadata.txt (github.com/numediart/UMONS-TAICHI)


def functional_series(P, groundY=0.0):
    """per frame from joint positions [n,>=22,3] (SMPL-X order: 1/2 hips, 4/5 knees, 7/8 ankles, 20/21 wrists):
    wrist height (mean of both, above groundY), knee flexion (mean of both, deg), pelvis yaw (hip line, rad, absolute)"""
    wh = (P[:, 20, 1] + P[:, 21, 1]) / 2 - groundY
    def flex(h, k, a):
        u = P[:, h] - P[:, k]; v = P[:, a] - P[:, k]
        c = np.sum(u * v, 1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)
        return 180 - np.degrees(np.arccos(np.clip(c, -1, 1)))
    kf = (flex(1, 4, 7) + flex(2, 5, 8)) / 2
    lat = P[:, 1] - P[:, 2]; yaw = np.arctan2(-lat[:, 2], lat[:, 0])
    return wh, kf, yaw


def rel_yaw_deg(yaw):
    m = np.angle(np.mean(np.exp(1j * yaw)))
    return np.degrees(np.angle(np.exp(1j * (yaw - m))))


def cmd_bands(a):
    """across-performer mean +- 1 SD of wrist height, knee flexion and pelvis yaw (UMONS Qualisys trials of the same class,
    one trial per participant, time-normalised), phase-aligned to each recorded segment of a sequence; written into the
    sequence meta as "bands" (pack header) for the Home 01 / B functional panel"""
    z = np.load(a.seq, allow_pickle=False)
    segs = json.loads(str(z["segments"])); meta = json.loads(str(z["meta"])); gender = str(z["gender"])
    M = np.load(os.path.join(MODELS, f"SMPLX_{gender}.npz"), allow_pickle=True); vt = M["v_template"]
    model_h = float(vt[:, 1].max() - vt[:, 1].min())
    trials = open(a.list).read().strip().split(",")
    ph = np.linspace(0, 1, 101); per_class = {}
    for tid in trials:
        G = tid[9:12]
        if G not in {s_["id"] for s_ in segs}: continue
        fs, names, X = load_qualisys(os.path.join(a.tsv_dir, tid + ".npz"))
        tt, L, _ = qualisys_landmarks(fs, names, X, 2.0)
        Pj = np.zeros((len(tt), 22, 3))
        for j, k in ((1, "hipL"), (2, "hipR"), (4, "kneeL"), (5, "kneeR"), (7, "ankleL"), (8, "ankleR"), (20, "wristL"), (21, "wristR")):
            Pj[:, j] = L[k]
        wh, kf, yaw = functional_series(Pj, 0.0)
        wh = wh * model_h / (STATURE_CM[tid[:3]] / 100.0)
        yd = rel_yaw_deg(yaw); g = np.linspace(0, 1, len(tt))
        per_class.setdefault(G, []).append((tid[:3], np.interp(ph, g, wh), np.interp(ph, g, kf), np.interp(ph, g, yd)))
    F = len(z["Q"]); step = a.step
    out = {k: {"mean": [None] * ((F + step - 1) // step), "sd": [None] * ((F + step - 1) // step)} for k in ("wristHeight", "kneeFlexion", "pelvisYaw")}
    nperf = {}
    for sg in segs:
        rows = per_class.get(sg["id"], [])
        if not rows: continue
        nperf[sg["id"]] = len({r[0] for r in rows})
        A = {k: np.stack([r[i + 1] for r in rows]) for i, k in enumerate(("wristHeight", "kneeFlexion", "pelvisYaw"))}
        head, tail = sg.get("trimmed_frames_head_tail", [0, 0]); n = sg["trial_frames"]; nu = n + head + tail
        t0 = sg.get("trial_start", sg["start"] + sg["transition_frames"]); t1 = sg.get("trial_end", sg["end"])
        off = sg.get("trial_phase_start_frame", 0)
        for f in range(t0, t1 + 1):
            if f % step: continue
            p = (head + off + f - t0) / (nu - 1)
            for k, X in A.items():
                v = np.array([np.interp(p, ph, x) for x in X])
                out[k]["mean"][f // step] = round(float(v.mean()), 4 if k == "wristHeight" else 2)
                out[k]["sd"][f // step] = round(float(v.std(ddof=1)), 4 if k == "wristHeight" else 2)
    bands = {"step": step, "series": out, "units": {"wristHeight": "m", "kneeFlexion": "deg", "pelvisYaw": "deg"},
             "performers": nperf,
             "note": ("across-performer mean and SD (one UMONS-TAICHI Qualisys trial per participant and form, D01, time-normalised and "
                      "aligned to each recorded segment; null in the transitions). wristHeight = mean of both wrist centres above the floor, "
                      "scaled by model stature / performer stature; kneeFlexion = mean of both knees, 180 - hip-knee-ankle angle; "
                      "pelvisYaw = hip-line facing relative to the trial's mean facing. Compute the same from the pack's joints "
                      "(wrists 20/21 above groundY; hips 1/2, knees 4/5, ankles 7/8).")}
    meta["bands"] = bands
    out_ = {k: z[k] for k in z.files}; out_["meta"] = json.dumps(meta, ensure_ascii=False)
    np.savez_compressed(a.out, **out_)
    print(f"bands: {nperf}; samples every {step} frames; {sum(v is not None for v in out['wristHeight']['mean'])} of {len(out['wristHeight']['mean'])} filled")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit")
    f.add_argument("--trial", required=True); f.add_argument("--source", choices=["kinect", "qualisys"], required=True)
    f.add_argument("--gender", default="NEUTRAL", choices=["NEUTRAL", "FEMALE", "MALE"])
    f.add_argument("--kinect-cache", default=""); f.add_argument("--tsv-dir", default="")
    f.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoothing of the 30 fps targets (frames)")
    f.add_argument("--iters", type=int, default=700); f.add_argument("--w-smooth", type=float, default=4.0)
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_fit)
    m = sp.add_parser("motion", help="fit npz -> root + contacts + leg IK (motion npz)")
    m.add_argument("--fit", required=True); m.add_argument("--out", required=True); m.add_argument("--w-y", type=float, default=0.3)
    m.add_argument("--ik", action="store_true", help="two-bone IK + floor pass instead of the torch polish")
    m.add_argument("--polish-iters", type=int, default=1000)
    m.add_argument("--w-dsmooth", type=float, default=3e4); m.add_argument("--soft-sigma", type=float, default=2.0)
    m.add_argument("--gaze", type=float, default=-2.0, help="head elevation in quiet frames (deg); pass nan to skip")
    m.set_defaults(fn=cmd_motion)
    s1 = sp.add_parser("single", help="one motion npz -> one-segment sequence npz")
    s1.add_argument("--motion", required=True); s1.add_argument("--out", required=True); s1.add_argument("--loop", default="cycle")
    s1.set_defaults(fn=cmd_single)
    q = sp.add_parser("sequence", help="motion npz list (play order) -> closed loop with planned transition steps")
    q.add_argument("--motions", required=True); q.add_argument("--out", required=True)
    q.add_argument("--step-frames", type=int, default=54); q.add_argument("--blend-frames", type=int, default=36)
    q.add_argument("--trim-max", type=float, default=0.6, help="max quiet head/tail trimmed per trial (s)")
    q.add_argument("--floor-pass", action="store_true")
    q.set_defaults(fn=cmd_sequence)
    po = sp.add_parser("polish", help="whole-loop torch cleanup of a sequence npz (contacts, floor, smoothness)")
    po.add_argument("--seq", required=True); po.add_argument("--out", required=True)
    po.add_argument("--iters", type=int, default=1000); po.add_argument("--w-pose", type=float, default=1.0)
    po.add_argument("--lr", type=float, default=0.002)
    po.add_argument("--w-dsmooth", type=float, default=3e4); po.add_argument("--soft-sigma", type=float, default=2.0)
    po.add_argument("--w-leg", type=float, default=0.15); po.add_argument("--w-smooth", type=float, default=30.0)
    po.set_defaults(fn=cmd_polish)
    bd = sp.add_parser("bands", help="card B: across-performer bands into a sequence npz's meta")
    bd.add_argument("--seq", required=True); bd.add_argument("--out", required=True); bd.add_argument("--list", required=True)
    bd.add_argument("--tsv-dir", required=True); bd.add_argument("--step", type=int, default=3)
    bd.set_defaults(fn=cmd_bands)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
