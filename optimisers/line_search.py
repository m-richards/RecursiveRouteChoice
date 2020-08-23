"""
% More and Thuente Arc Search
% This is a port of Tien Mai's  matlab code which is
a modified implementation of the algorithm documented in:
%
% J. J. More and D. J. Thuente. Line Search Algorithms with Guaranteed
% Sufficient Decrease. TOMS 20-3. September 1994. pg 286-307.
"""
import numpy as np
from enum import Enum, auto

INITIAL_STEP_LENGTH = 1.0
NEGATIVE_CURVATURE_PARAMETER = 0.0
SUFFICIENT_DECREASE_PARAMETER = 0.0001
CURVATURE_CONDITION_PARAMETER = 0.9
X_TOLERENT = 2.2e-16
MINIMUM_STEP_LENGTH = 0
MAXIMUM_STEP_LENGTH = 1000
MAX_FEV = 10  # Maximum number of function evaluations


class LineSearchFlags(Enum):
    TERMINATION_ROUNDING_ERROR = auto()
    TERMINATION_STPMAX = auto()
    TERMINATION_STPMIN = auto()
    TERMINATION_INTERVAL_TOO_SMALL = auto()
    TERMINATION_MAX_FUNC_EVALS = auto()
    TERMINATION_STRONG_WOLFE_MET = auto()
    TERMINATION_STRONG_WOLFE_AND_STPMAX = auto()


def line_search_astep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    """
    % This function computes a safeguarded step for a search
    % procedure and updates an interval that contains a step that
    % satisfies a sufficient decrease and a curvature condition.
    %
    % The parameter stx contains the step with the least function
    % value. If brackt is set to true (1) then a minimizer has
    % been bracketed in an interval with endpoints stx and sty.
    % The parameter stp contains the current step.
    % The subroutine assumes that if brackt is set to true then
    %
    % min(stx,sty) < stp < max(stx,sty),
    %
    % and that the derivative at stx is negative in the direction
    % of the step.
    %
    % The subroutine statement is
    %
    % stf = line_search_astep(stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,stpmin,stpmax)
    %
    % where
    %
    % stx is a double precision variable.
    % On entry stx is the best step obtained so far and is an
    % endpoint of the interval that contains the minimizer.
    %
    % fx is a double precision variable.
    % On entry fx is the function at stx.
    %
    % dx is a double precision variable.
    % On entry dx is the derivative of the function at
    % stx. The derivative must be negative in the direction of
    % the step, that is, dx and stp - stx must have opposite
    % signs.
    %
    % sty is a double precision variable.
    % On entry sty is the second endpoint of the interval that
    % contains the minimizer.
    %
    % fy is a double precision variable.
    % On entry fy is the function at sty.
    %
    % dy is a double precision variable.
    % On entry dy is the derivative of the function at sty.
    %
    % stp is a double precision variable.
    % On entry stp is the current step. If brackt is set to true
    % then on input stp must be between stx and sty.
    %
    % fp is a double precision variable.
    % On entry fp is the function at stp
    %
    % dp is a double precision variable.
    % On entry dp is the the derivative of the function at stp.
    %
    % brackt is an logical variable.
    % On entry brackt specifies if a minimizer has been bracketed.
    % Initially brackt must be set to .false.
    %
    % stpmin is a double precision variable.
    % On entry stpmin is a lower bound for the step.
    %
    % stpmax is a double precision variable.
    % On entry stpmax is an upper bound for the step.
    %
    % MINPACK-1 Project. June 1983
    % Argonne National Laboratory.
    % Jorge J. More' and David J. Thuente.
    %
    % MINPACK-2 Project. November 1993.
    % Argonne National Laboratory and University of Minnesota.
    % Brett M. Averick and Jorge J. More'.
    :return: 
    :rtype: 
    """
    # print("\tInput astep:", stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax)
    # parameter
    p66 = 0.66  # TODO why a magic constant

    sgnd = dp * (dx / abs(dx))

    if (fp > fx):
        # First case: A higher function value. The minimum is bracketed.
        # If the cubic step is closer to stx than the quadratic step, the
        # cubic step is taken, otherwise the average of the cubic and
        # quadratic steps is taken.

        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = max((abs(theta), abs(dx), abs(dp)))
        gamma = s * np.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if (stp < stx):
            gamma = -gamma

        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx)
        if (abs(stpc - stx) < abs(stpq - stx)):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / 2.0

        # brackt = true

    elif (sgnd < 0.0):
        # Second case: A lower function value and derivatives of opposite
        # sign. The minimum is bracketed. If the cubic step is farther from
        # stp than the secant step, the cubic step is taken, otherwise the
        # secant step is taken.

        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = max((abs(theta), abs(dx), abs(dp)))
        gamma = s * np.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if (stp > stx):
            gamma = -gamma

        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if (abs(stpc - stp) > abs(stpq - stp)):
            stpf = stpc
        else:
            stpf = stpq

        # brackt = true

    elif (abs(dp) < abs(dx)):
        # Third case: A lower function value, derivatives of the same sign,
        # and the magnitude of the derivative decreases.

        # The cubic step is computed only if the cubic ts to infinity
        # in the direction of the step or if the minimum of the cubic
        # is beyond stp. Otherwise the cubic step is defined to be the
        # secant step.

        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = max((abs(theta), abs(dx), abs(dp)))

        # The case gamma = 0 only arises if the cubic does not t
        # to infinity in the direction of the step.

        gamma = s * np.sqrt(max(0.0, (theta / s) ** 2 - (dx / s) * (dp / s)))
        if (stp > stx):
            gamma = -gamma

        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        if (r < 0.0 and gamma != 0.0):
            stpc = stp + r * (stx - stp)
        elif (stp > stx):
            stpc = stpmax
        else:
            stpc = stpmin

        stpq = stp + (dp / (dp - dx)) * (stx - stp)

        if (brackt):

            # A minimizer has been bracketed. If the cubic step is
            # closer to stp than the secant step, the cubic step is
            # taken, otherwise the secant step is taken.

            if (abs(stpc - stp) < abs(stpq - stp)):
                stpf = stpc
            else:
                stpf = stpq

            if (stp > stx):
                stpf = min(stp + p66 * (sty - stp), stpf)
            else:
                stpf = max(stp + p66 * (sty - stp), stpf)

        else:

            # A minimizer has not been bracketed. If the cubic step is
            # farther from stp than the secant step, the cubic step is
            # taken, otherwise the secant step is taken.

            if (abs(stpc - stp) > abs(stpq - stp)):
                stpf = stpc
            else:
                stpf = stpq

            stpf = min(stpmax, stpf)
            stpf = max(stpmin, stpf)

    else:
        # Fourth case: A lower function value, derivatives of the same sign,
        # and the magnitude of the derivative does not decrease. If the
        # minimum is not bracketed, the step is either stpmin or stpmax,
        # otherwise the cubic step is taken.

        if (brackt):
            theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
            s = max((abs(theta), abs(dy), abs(dp)))
            gamma = s * np.sqrt((theta / s) ** 2 - (dy / s) * (dp / s))
            if (stp > sty):
                gamma = -gamma

            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p / q
            stpc = stp + r * (sty - stp)
            stpf = stpc
        elif (stp > stx):
            stpf = stpmax
        else:
            stpf = stpmin
    return stpf


"""
% More and Thuente Arc Search
% This is a port of Tien Mai's  matlab code which is 
a modified implementation of the algorithm documented in:
%
% J. J. More and D. J. Thuente. Line Search Algorithms with Guaranteed
% Sufficient Decrease. TOMS 20-3. September 1994. pg 286-307.
%
% It attempts to find a step length stp such that:
%
% Sufficient decrease is met:
% f(x+s) <= f(x) + ftol*(ginit*stp + 0.5*min(ncur,0)*stp^2)
%
% Curvature condition is met:
% |g(x+s)| <= gtol*|ginit + min(ncur,0)*stp|
%
% s = arc(stp) is the displacement vector.
% ginit the first derivative along the search arc at stp=0.
%
% It makes the assumption that ftol <= gtol and thus does not require the
% modified interval updating rules. It also uses bisection to compute the
% next trial step instead of polynomial interpolation.
%
% This version uses a function of a vector and is efficient with
% evaluations.
%
% Input:
% fcn = a handle to a function that returns the value and first
% derivative. Usage in code: [f g] = fcn(x)
% x = current point
% f = function value at x
% g = gradient at x
% arc = search arc function. Usage in code: [s ds] = arc(stp)
% stp = initial step length
% ncur = negative curvature parameter
% ftol = sufficient decrease parameter, mu in the paper
% gtol = curvature condition parameter, eta in the paper
% xtol = the algorithm terminates if the width of the interval is less
% than xtol.
% stpmin = minimum step length allowed. It is acceptable to set stpmin=0.
% stpmax = maximum step length allowed.
% maxfev = maximum number of function evaluations
% fid = file identifier for test output (optional)
%
% Output:
% x = final point
% f = final function value
% g = final gradient
% stp = step length
% info = termination flag
% nfev = number of function evaluations.
%
% Termination Flags:
% info = 1 if stp satisfies the descent and curvature condition
% info = 2 if interval size is less than xtol
% info = 3 if algorithm has exceeded maxfev
% info = 4 if stpmin > 0 and stp == stpmin
% info = 5 if stp == stpmax
% info = 6 if stp == stpmax & strong wolfe conditions met
% info = 7 if rounding errors prevent progress
"""


# note, maxfev has moved up in order
def line_search_asrch(fcn, x, f, g, arc, stp, maxfev,
                      ncur=NEGATIVE_CURVATURE_PARAMETER,
                      ftol=SUFFICIENT_DECREASE_PARAMETER,
                      gtol=CURVATURE_CONDITION_PARAMETER,
                      xtol=X_TOLERENT, stpmin=MINIMUM_STEP_LENGTH,
                      stpmax=MAXIMUM_STEP_LENGTH, print_flag=True, fname=None,
                      bisect=0.0,
                      debug_counter=None):
    """outputs [x f g stp info_out_flag nfev] =
    % list of variables and parameters
    % extrap = parameter for extrapolations
    % is_bracketed = true once bracket containing solution is found
    % info_out_flag = status flag for output
    % nfev = number of function evaluations
    % s = displacement vector from arc
    % ds = derivative of displacement vector from arc
    %
    % stx = step size at "l" point
    % fx = function value at "l" point
    % dx = derivative of search function at "l" point
    %
    % sty = step size at "u" point
    % fy = function value at "u" point
    % dy = derivative of search function at "u" point
    %
    % stp = trial step size
    % fp = function value at trial step size
    % dp = derivative of search function at trial step size
    %
    % mfx = modified function value at "l" point
    % mdx = modified derivative value at "l" point
    %
    % mfp = modified function value at trial point
    % mdp = modified derivative value at trial point
    %
    % Note al and au define the bounds of the bracket if one is found. al and au
    % are the endpoints of the bracket, but are not ordered.
    %
    % finit = initial function value
    % ginit = initial gradient
    % amin = minimium step size
    % amax = maximum step size
    % ucase = arc search update case
    """
    g_start = g
    # parameters
    xtrapu = 4
    p66 = 0.66

    # flags
    is_bracketed = False
    info_out_flag = False
    # counters
    nfev = 0

    # interval width tracker
    width = stpmax - stpmin
    width1 = 2 * width

    # inital values
    [s, ds] = arc(0)

    stx = 0.0
    fx = f
    dx = np.dot(g, ds)

    finit = fx
    ginit = dx

    fp = 0.0
    dp = 0.0

    sty = 0.0
    fy = 0.0
    dy = 0.0

    # formatting & printing
    if print_flag or fname is not None:
        print_flag = True
        header_format = "{:4} {:6}" + 5 * " {:14.14}" + "|{:4.4}\n"
        data_format_1 = "{:4} {:6}" + 5 * " {:14.8g}"
        data_format_2 = "|{:4}\n"
        print(header_format.format("nfev", "b", "stx", "sty", "stp", "fp", "dp", "case",
                                   file=fname), end="")
    n = 0
    while True:
        print(f"count: {debug_counter}:{n}")
        n += 1
        if is_bracketed:
            stmin = min(stx, sty)
            stmax = max(stx, sty)
        else:
            stmin = stx
            stmax = stp + xtrapu * (stp - stx)

        # safeguard the trial step size (make sure step passed in is in legal bounds
        stp = max(stp, stpmin)
        stp = min(stp, stpmax)
        # print("stp = ", stp)

        # If an unusual termination is to occur then let
        # stp be the lowest point obtained so far.
        if ((is_bracketed and (stp <= stmin or stp >= stmax))
                or nfev >= maxfev - 1
                or (is_bracketed and stmax - stmin <= xtol * stmax)):
            stp = stx
        (s, ds) = arc(stp)
        # print("(s, ds) = ", s, ds)
        # Likelihood at new beta
        f, g = fcn(x + s)
        # print("\t g= ", np.all(g==g_start), g, g_start)

        fp = float(f)
        dp = np.dot(g, ds)
        nfev += 1

        if print_flag:
            print(data_format_1.format(nfev, is_bracketed, stx, sty, stp, fp, dp), file=fname,
                  end="")

        # compute modified function values
        mstx = stx
        mfx = fx - finit - ftol * (ginit * stx + 0.5 * min(ncur, 0) * stx ** 2)
        mdx = dx - ftol * (ginit + min(ncur, 0) * stx)

        # mstp = stp
        mfp = fp - finit - ftol * (ginit * stp + 0.5 * min(ncur, 0) * stp ** 2)
        mdp = dp - ftol * (ginit + min(ncur, 0) * stp)

        msty = sty
        mfy = fy - finit - ftol * (ginit * sty + 0.5 * min(ncur, 0) * sty ** 2)
        mdy = dy - ftol * (ginit + min(ncur, 0) * sty)

        # convergence tests TODO could wrap this up to not repeat myself - not worth for now
        #  since don't have understanding

        # terminate if rounding errors prevent progress
        if is_bracketed and (stp <= stmin or stp >= stmax):
            info_out_flag = LineSearchFlags.TERMINATION_ROUNDING_ERROR

        # terminate at stpmax
        if stp == stpmax and mfp <= 0 and mdp < 0:
            info_out_flag = LineSearchFlags.TERMINATION_STPMAX

        # terminate at stpmin
        if stpmin > 0 and stp == stpmin and (mfp > 0 or mdp >= 0):
            info_out_flag = LineSearchFlags.TERMINATION_STPMIN

        # terminate if interval is too small
        if is_bracketed and (stmax - stmin < xtol * stmax):
            info_out_flag = LineSearchFlags.TERMINATION_INTERVAL_TOO_SMALL

        # terminate if reached maximum number of function evaluations
        if nfev >= maxfev:
            info_out_flag = LineSearchFlags.TERMINATION_MAX_FUNC_EVALS

        # terminate if strong wolfe conditions are met
        if (fp <= finit + ftol * (ginit * stp + 0.5 * min(ncur, 0) * stp ** 2)
                and abs(dp) <= gtol * abs(ginit + min(ncur, 0) * stp)):
            info_out_flag = LineSearchFlags.TERMINATION_STRONG_WOLFE_MET

        # if strong wolfe conditions are met with at == stpmax
        if info_out_flag == LineSearchFlags.TERMINATION_STRONG_WOLFE_MET and stp == stpmax:
            info_out_flag = LineSearchFlags.TERMINATION_STRONG_WOLFE_AND_STPMAX

        if info_out_flag is not False:  # if we have info_out_flag
            x = x + s
            if print_flag:
                print(f"|t-{LineSearchFlags(info_out_flag).name}", file=fname)
            return (x, f, g, stp, info_out_flag, nfev)

        # update the interval
        if mfp > mfx:
            # case U1
            # stx = stx fx = fx dx = dx
            # tODO these values look static per iter sometimes
            sty = stp
            fy = fp
            dy = dp
            is_bracketed = True
            ucase = 1
        elif mfp <= mfx and mdp * (stx - stp) > 0:
            # case U2
            stx = stp
            fx = fp
            dx = dp
            # sty = sty fy = fy dy = dy
            ucase = 2
        else:  # mfp <= mfx && mdp*(stx-stp) < 0
            # case U3
            sty = stx
            fy = fx
            dy = dx
            stx = stp
            fx = fp
            dx = dp
            is_bracketed = True
            ucase = 3

        # print the case
        if print_flag:
            print(f"|u-{ucase}")
            # compute new trial step size
        if is_bracketed and bisect:
            # bisect if desired
            stp = stx + 0.5 * (sty - stx)
        else:
            # compute new step using interpolation
            stp = line_search_astep(
                mstx, mfx, mdx, msty, mfy, mdy, stp, mfp,
                mdp, is_bracketed, stmin, stmax)
            # print("interp step", stp)
            # safeguard the step and update the interval width tracker
        if is_bracketed:
            if (abs(sty - stx) >= p66 * width1):
                stp = stx + 0.5 * (sty - stx)

            width1 = width
            width = abs(sty - stx)
