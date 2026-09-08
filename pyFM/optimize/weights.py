import numpy as np

from .base_functions import energy_func_std, oplist_commutation

__all__ = [
    "resolvent_mask",
    "effective_p",
    "canonical_taus",
    "legacy_taus",
]


def _oplist_trace(op_list, skew=False):
    """``sum_i k2||M1_i||_F^2 + k1||M2_i||_F^2 - 2 tr M1_i tr M2_i``, clamped at 0.

    ``skew=True`` ignores the cross term.
    """
    if len(op_list) == 0:
        return 0.0, 0.0

    total = 0.0
    positive = 0.0
    for op1, op2 in zip(*op_list):
        k1, k2 = op1.shape[0], op2.shape[0]
        pos = k2 * np.sum(np.square(op1)) + k1 * np.sum(np.square(op2))
        positive += pos
        total += pos if skew else pos - 2 * np.trace(op1) * np.trace(op2)

    return float(max(total, 0.0)), float(positive)


def resolvent_mask(evals1, evals2, gamma=0.5):
    """
    LBO Mask, Ren et al., SGP 2019.

    The standard ``(lambda_i - mu_j)^2`` mask is unbounded: the energy diverges in ``k``
    even for ``C = Id``, so no choice of ``w_lap`` makes the term well posed. Any bounded
    ``g(Delta)`` has the same null space; this uses the complex resolvent
    ``(Delta^gamma - i)^-1``. Both spectra are divided by the same factor,
    ``max(max Lambda_1, max Lambda_2)``, so the shapes stay comparable.

    Parameters
    ----------
    evals1, evals2 : (K1,), (K2,) np.ndarray
    gamma : float
        In (0, 1]. Larger narrows the funnel, i.e. a stronger isometric prior.

    Returns
    -------
    mask : (K2, K1) np.ndarray
        Rows index mesh2, columns index mesh1 -- the orientation of ``C``.
    """
    scale = max(np.max(np.abs(evals1)), np.max(np.abs(evals2)))

    lam = np.abs(evals1) / scale
    mu = np.abs(evals2) / scale

    lam_g, mu_g = lam**gamma, mu**gamma

    re1, re2 = lam_g / (lam_g**2 + 1), mu_g / (mu_g**2 + 1)
    im1, im2 = 1 / (lam_g**2 + 1), 1 / (mu_g**2 + 1)

    return np.square(re1[None, :] - re2[:, None]) + np.square(im1[None, :] - im2[:, None])


def effective_p(*descr_red):
    """
    Estimation of independent descriptors using gram matrix.

    ``p_eff = (tr G)^2 / ||G||_F^2`` with ``G = F.T @ F``.
    Ignores the constant mode for each descriptor.

    Parameters
    ----------
    *descr_red : (K, p) np.ndarray
        Descriptors in the reduced basis, one array per shape.

    Returns
    -------
    p_eff : float
    """
    values = []
    for F in descr_red:
        gram = F[1:].T @ F[1:]
        denom = np.sum(np.square(gram))
        values.append(np.square(np.trace(gram)) / denom if denom > 0 else 1.0)
    return float(np.mean(values)) if values else 1.0


def canonical_taus(descr1_red, descr2_red, descr_op, orient_op, lap_mask, p_eff=None):
    """
    There are four normalizers.

        descr   tau = (m / k1) * ||F1||_F^2                                  / p_eff
        lap     tau = (m / (k1*k2)) * sum_ij M_ij
        dcomm   tau = (m / (k1*k2)) * (k2||M1||^2 + k1||M2||^2 - 2 tr tr)    / p_eff
        orient  tau = (m / (k1*k2)) * (k2||W1||^2 + k1||W2||^2)              / p_eff

    Using centered descriptors will help.

    Parameters
    ----------
    descr1_red : (k1, p) np.ndarray
    descr2_red : (k2, p) np.ndarray
    descr_op : tuple of np.ndarray or []
        ``(ops1, ops2)`` stacked as (p, k1, k1) and (p, k2, k2).
    orient_op : tuple of np.ndarray or []
    lap_mask : (k2, k1) np.ndarray
        Raw LBO mask.
    p_eff : float, optional
        Defaults to :func:`effective_p` of the two reduced descriptor banks.

    Returns
    -------
    taus : tuple of float
        ``(tau_descr, tau_lap, tau_dcomm, tau_orient)``. A degenerate term gets 1.0.
    """

    k1 = descr1_red.shape[0]
    k2 = descr2_red.shape[0]
    m = min(k1, k2)
    if p_eff is None:
        p_eff = effective_p(descr1_red, descr2_red)
    p_eff = max(p_eff, 1e-12)

    def _guard(value):
        return value if value > 0 else 1.0

    def _oplist_tau(op_list, skew):
        total, positive = _oplist_trace(op_list, skew=skew)
        if total <= 1e-12 * positive:
            return 1.0
        return total * m / (k1 * k2) / p_eff

    # Ignore the constant mode for the descriptor
    tau_descr = _guard(m * np.sum(np.square(descr1_red[1:])) / k1 / p_eff)
    tau_lap = _guard(m * float(np.sum(lap_mask)) / (k1 * k2))
    tau_descr_op = _oplist_tau(descr_op, skew=False)
    tau_orient_op = _oplist_tau(orient_op, skew=True)

    return (
        tau_descr,
        tau_lap,
        tau_descr_op,
        tau_orient_op,
    )


def legacy_taus(
    K,
    w_descr,
    w_lap,
    w_dcomm,
    w_orient,
    descr1_red,
    descr2_red,
    descr_op,
    orient_op,
    lap_mask,
    verbose=False,
):

    if np.issubdtype(type(K), np.integer):
        k1 = k2 = K
    else:
        k1, k2 = K

    tau_orient = 1
    if w_orient > 0:
        C_eye = np.eye(k2, k1)
        w_others = w_descr + w_lap + w_dcomm
        w_others = w_others if w_others > 0 else 1.0
        eval_native = energy_func_std(
            C_eye,
            w_descr / w_others,
            w_lap / w_others,
            w_dcomm / w_others,
            0,
            descr1_red,
            descr2_red,
            descr_op,
            orient_op,
            lap_mask / lap_mask.sum(),
        )
        eval_orient = oplist_commutation(C_eye, orient_op)
        if eval_orient > 0:
            scale = eval_native / eval_orient
            # w_orient *= scale
            tau_orient = 1 / scale
            if verbose:
                print(f"\tOrientation weight scaled by {scale:.2e}")
        elif verbose:
            print("\tOrientation operator has zero energy; skipping orientation weight rescaling")

    w_total = w_descr + w_lap + w_dcomm + w_orient / tau_orient
    w_total = w_total if w_total > 0 else 1.0

    tau_descr = w_total
    tau_lap = w_total * np.sum(lap_mask)
    tau_dcomm = w_total
    tau_orient = tau_orient * w_total

    return tau_descr, tau_lap, tau_dcomm, tau_orient
