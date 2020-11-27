import numpy as np


def descr_preservation(C, descr1_red, descr2_red):
    """
    Compute the descriptor preservation constraint

    Parameters
    ---------------------
    C      : (K2,K1) Functional map
    descr1 : (K1,p) descriptors on first basis
    descr2 : (K2,p) descriptros on second basis

    Output
    ---------------------
    energy : descriptor preservation squared norm
    """
    return 0.5 * np.square(C @ descr1_red - descr2_red).sum()


def descr_preservation_grad(C, descr1_red, descr2_red):
    """
    Compute the gradient of the descriptor preservation constraint

    Parameters
    ---------------------
    C      : (K2,K1) Functional map
    descr1 : (K1,p) descriptors on first basis
    descr2 : (K2,p) descriptros on second basis

    Output
    ---------------------
    gradient : gradient of the descriptor preservation squared norm
    """
    return (C @ descr1_red - descr2_red) @ descr1_red.T


def LB_commutation(C, ev_sqdiff):
    """
    Compute the LB commutativity constraint

    Parameters
    ---------------------
    C      : (K2,K1) Functional map
    ev_sqdiff : (K2,K1) [normalized] matrix of squared eigenvalue differences

    Output
    ---------------------
    energy : (float) LB commutativity squared norm
    """
    return 0.5 * (np.square(C) * ev_sqdiff).sum()


def LB_commutation_grad(C, ev_sqdiff):
    """
    Compute the gradient of the LB commutativity constraint

    Parameters
    ---------------------
    C         : (K2,K1) Functional map
    ev_sqdiff : (K2,K1) [normalized] matrix of squared eigenvalue differences

    Output
    ---------------------
    gradient : (K2,K1) gradient of the LB commutativity squared norm
    """
    return C * ev_sqdiff


def op_commutation(C, op1, op2):
    """
    Compute the operator commutativity constraint.
    Can be used with descriptor multiplication operator

    Parameters
    ---------------------
    C   : (K2,K1) Functional map
    op1 : (K1,K1) operator on first basis
    op2 : (K2,K2) descriptros on second basis

    Output
    ---------------------
    energy : (float) operator commutativity squared norm
    """
    return 0.5 * np.square(C @ op1 - op2 @ C).sum()


def op_commutation_grad(C, op1, op2):
    """
    Compute the gradient of the operator commutativity constraint.
    Can be used with descriptor multiplication operator

    Parameters
    ---------------------
    C   : (K2,K1) Functional map
    op1 : (K1,K1) operator on first basis
    op2 : (K2,K2) descriptros on second basis

    Output
    ---------------------
    gardient : (K2,K1) gradient of the operator commutativity squared norm
    """
    return op2.T @ (op2 @ C - C @ op1) - (op2 @ C - C @ op1) @ op1.T


def oplist_commutation(C, op_list):
    """
    Compute the operator commutativity constraint for a list of pairs of operators
    Can be used with a list of descriptor multiplication operator

    Parameters
    ---------------------
    C   : (K2,K1) Functional map
    op_list : list of tuple( (K1,K1), (K2,K2) ) operators on first and second basis

    Output
    ---------------------
    energy : (float) sum of operators commutativity squared norm
    """
    energy = 0
    for (op1, op2) in op_list:
        energy += op_commutation(C, op1, op2)

    return energy


def oplist_commutation_grad(C, op_list):
    """
    Compute the gradient of the operator commutativity constraint for a list of pairs of operators
    Can be used with a list of descriptor multiplication operator

    Parameters
    ---------------------
    C   : (K2,K1) Functional map
    op_list : list of tuple( (K1,K1), (K2,K2) ) operators on first and second basis

    Output
    ---------------------
    gradient : (K2,K1) gradient of the sum of operators commutativity squared norm
    """
    gradient = 0
    for (op1, op2) in op_list:
        gradient += op_commutation_grad(C, op1, op2)
    return gradient


def energy_func_std(C, descr_mu, lap_mu, descr_comm_mu, orient_mu, descr1_red, descr2_red, list_descr, orient_op, ev_sqdiff):
    """
    Evaluation of the energy for standard FM computation

    Parameters:
    ----------------------
    C               : (K2*K1) or (K2,K1) Functional map
    descr_mu        : scaling of the descriptor preservation term
    lap_mu          : scaling of the laplacian commutativity term
    descr_comm_mu   : scaling of the descriptor commutativity term
    orient_mu       : scaling of the orientation preservation term
    descr1          : (K1,p) descriptors on first basis
    descr2          : (K2,p) descriptros on second basis
    list_descr      : p-uple( (K1,K1), (K2,K2) ) operators on first and second basis
                      related to descriptors.
    orient_op       : p-uple( (K1,K1), (K2,K2) ) operators on first and second basis
                      related to orientation preservation operators.
    ev_sqdiff       : (K2,K1) [normalized] matrix of squared eigenvalue differences

    Output
    ------------------------
    energy : float - value of the energy
    """
    k1 = descr1_red.shape[0]
    k2 = descr2_red.shape[0]
    C = C.reshape((k2,k1))

    energy = 0

    if descr_mu > 0:
        energy += descr_mu * descr_preservation(C, descr1_red, descr2_red)

    if lap_mu > 0:
        energy += lap_mu * LB_commutation(C, ev_sqdiff)

    if descr_comm_mu > 0:
        energy += descr_comm_mu * oplist_commutation(C, list_descr)

    if orient_mu > 0:
        energy += orient_mu * oplist_commutation(C, orient_op)

    return energy


def grad_energy_std(C, descr_mu, lap_mu, descr_comm_mu, orient_mu, descr1_red, descr2_red, list_descr, orient_op, ev_sqdiff):
    """
    Evaluation of the gradient of the energy for standard FM computation

    Parameters:
    ----------------------
    C               : (K2*K1) or (K2,K1) Functional map
    descr_mu        : scaling of the descriptor preservation term
    lap_mu          : scaling of the laplacian commutativity term
    descr_comm_mu   : scaling of the descriptor commutativity term
    orient_mu       : scaling of the orientation preservation term
    descr1          : (K1,p) descriptors on first basis
    descr2          : (K2,p) descriptros on second basis
    list_descr      : p-uple( (K1,K1), (K2,K2) ) operators on first and second basis
                      related to descriptors.
    orient_op       : p-uple( (K1,K1), (K2,K2) ) operators on first and second basis
                      related to orientation preservation operators.
    ev_sqdiff       : (K2,K1) [normalized] matrix of squared eigenvalue differences

    Output
    ------------------------
    gradient : (K2*K1) - value of the energy
    """
    k1 = descr1_red.shape[0]
    k2 = descr2_red.shape[0]
    C = C.reshape((k2,k1))

    gradient = np.zeros_like(C)

    if descr_mu > 0:
        gradient += descr_mu * descr_preservation_grad(C, descr1_red, descr2_red)

    if lap_mu > 0:
        gradient += lap_mu * LB_commutation_grad(C, ev_sqdiff)

    if descr_comm_mu > 0:
        gradient += descr_comm_mu * oplist_commutation_grad(C, list_descr)

    if orient_mu > 0:
        gradient += orient_mu * oplist_commutation_grad(C, orient_op)

    gradient[:,0] = 0
    return gradient.reshape(-1)