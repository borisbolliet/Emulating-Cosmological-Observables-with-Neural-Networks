#!/usr/bin/env python3
"""Batch generation of CLASS and CAMB observables for LHS-sampled cosmologies.

This script combines the sampling logic from ``LHC_Gen_YAML.py`` with the
observable calculation pipeline from ``Cl_CLASS_CAMB_Test.py``. It builds
100 batches of 2 cosmological parameter sets (configurable via CLI), computes
observables with both CLASS and CAMB for every successful draw, and stores all
intermediate and aggregated products on disk.
"""

import argparse
import json
import math
import os
from collections import OrderedDict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple
import sys

import numpy as np
import yaml
from astropy.cosmology import Planck15 as Planck
import time
import multiprocessing as mp
import tempfile
import pickle

import classy
import camb
from camb import CAMBparams, model
import pyDOE

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# --------------------------------------------------------------------------------------
# Shared cosmology constants (mirrors Cl_CLASS_CAMB_Test.py)
# --------------------------------------------------------------------------------------
h_planck = Planck.h


class LCDM:
    """Convenience container for Planck15 baseline parameters."""

    h = h_planck
    H0 = Planck.H0.value
    omega_b = Planck.Ob0 * h**2
    omega_cdm = (Planck.Om0 - Planck.Ob0) * h**2
    omega_k = Planck.Ok0
    Neff = Planck.Neff
    Tcmb = Planck.Tcmb0.value
    A_s = 2.097e-9
    tau_reio = 0.0540
    n_s = 0.9652
    m_ncdm = 0.06


# --------------------------------------------------------------------------------------
# Sampling configuration mirrors LHC_Gen_YAML.py
# --------------------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 1
DEFAULT_OUTPUT = Path("./sh_batch_outputs")

K_VALS = np.logspace(-4, 1, 200)
Z_BG = np.linspace(0, 5, 5_000)
L_MAX = 5_000
ERROR_LOG_FILENAME = "error_log.jsonl"

class SamplerConfig(object):
    """Container for execution parameters."""

    def __init__(self, yaml_file, n_batches, batch_size, output_dir, seed, chunk_size):
        self.yaml_file = yaml_file
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.seed = seed
        self.chunk_size = chunk_size


class ParamSpec(object):
    """Bounds and CLASS mapping for a sampled parameter."""

    def __init__(self, bounds, class_name):
        self.bounds = bounds
        self.class_name = class_name


# --------------------------------------------------------------------------------------
# YAML parsing and LHS sampling utilities
# --------------------------------------------------------------------------------------


def load_yaml_config(yaml_path: Path) -> Dict:
    with yaml_path.open("r") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def extract_param_specs(config: Dict) -> OrderedDict[str, ParamSpec]:
    """Extract parameter bounds directly from YAML file (dynamic list)."""
    params_section = config.get('params', {})
    specs: OrderedDict[str, ParamSpec] = OrderedDict()

    for name, entry in params_section.items():
        prior = entry.get('prior', {})
        if 'min' not in prior or 'max' not in prior:
            continue

        bounds = {'min': float(prior['min']), 'max': float(prior['max'])}
        specs[name] = ParamSpec(bounds=bounds, class_name=name)  # direct mapping

    return specs


def build_candidate_arrays(specs: OrderedDict[str, ParamSpec], n_samples: int) -> np.ndarray:
    arrays = []
    for spec in specs.values():
        bounds = spec.bounds
        arrays.append(np.linspace(bounds['min'], bounds['max'], n_samples))
    return np.vstack(arrays)


def generate_lhs_samples(
    specs: OrderedDict[str, ParamSpec],
    n_samples: int,
    seed: Optional[int] = None,
) -> List[Dict[str, float]]:
    if n_samples <= 0 or not specs:
        return []

    n_params = len(specs)
    param_arrays = build_candidate_arrays(specs, n_samples)

    if seed is not None:
        np.random.seed(seed)

    design = pyDOE.lhs(n_params, samples=n_samples, criterion=None)
    indices = (design * n_samples).astype(int)

    samples: List[Dict[str, float]] = []
    keys = list(specs.keys())
    for sample_idx in range(n_samples):
        values = {}
        for p_idx in range(n_params):
            values[keys[p_idx]] = param_arrays[p_idx][indices[sample_idx, p_idx]]
        
        if 'N_eff' in values and 'm_ncdm' in values:
            # Constants
            N_ncdm_const = 1
            deg_ncdm = 3

            # (Optional) store convenience keys
            values['N_ncdm'] = int(N_ncdm_const)
            values['deg_ncdm'] = int(deg_ncdm)
        
        if values['N_eff'] > 3.03959:
            values['N_ur'] = values['N_eff'] - 3.03959
            values['T_ncdm'] = 0.71611
        else:
            values['N_ur'] = 0.0
            values['T_ncdm'] = 0.71611 * (values['N_eff'] / 3.03959)**0.25

        samples.append(values)

    return samples


class LHSSamplePool:
    """Sample provider that replenishes LHS draws in configurable chunks."""

    def __init__(self, specs: OrderedDict[str, ParamSpec], rng: np.random.Generator, chunk_size: int) -> None:
        self._specs = specs
        self._rng = rng
        self._chunk_size = max(chunk_size, 1)
        self._buffer: Deque[Dict[str, float]] = deque()

    def _refill(self, n_samples: Optional[int] = None) -> None:
        target = n_samples or self._chunk_size
        seed = int(self._rng.integers(0, 2**32 - 1))
        new_samples = generate_lhs_samples(self._specs, target, seed=seed)
        self._buffer.extend(new_samples)

    def next(self) -> Dict[str, float]:
        if not self._buffer:
            self._refill()
        return self._buffer.popleft().copy()
    

def safe_compute(func, args, timeout=1800):
    """
    Run CLASS/CAMB in a separate process and save results to a temp file.
    Protects against segfaults and ensures large data are returned safely.
    """
    def target(q):
        try:
            result = func(*args)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
                q.put(("ok", f.name))
        except Exception as e:
            q.put(("error", repr(e)))

    q = mp.Queue()
    p = mp.Process(target=target, args=(q,))
    p.start()
    p.join(timeout)

    # Timeout handling
    if p.is_alive():
        p.kill()
        return None, f"Timeout after {timeout}s"

    # Collect results
    if not q.empty():
        status, payload = q.get()
        if status == "ok":
            filename = payload
            try:
                with open(filename, "rb") as f:
                    result = pickle.load(f)
            finally:
                os.remove(filename)
            return result, None
        else:
            return None, payload
    else:
        return None, "Unknown crash or segfault"


# --------------------------------------------------------------------------------------
# Parameter conversion and bookkeeping
# --------------------------------------------------------------------------------------


def logA_to_As(logA: float) -> float:
    """Convert log(10^10 A_s) to A_s."""
    return math.exp(logA) / 1e10


def build_full_parameter_dict(sample: Dict[str, float]) -> Dict[str, float]:
    """Combine sampled values with Planck baselines for downstream use."""
    full = {
        'logA': sample.get('logA', math.log(1e10 * LCDM.A_s)),
        'n_s': sample.get('n_s', LCDM.n_s),
        'H0': sample.get('H0', LCDM.H0),
        'omega_b': sample.get('omega_b', LCDM.omega_b),
        'omega_cdm': sample.get('omega_cdm', LCDM.omega_cdm),
        'tau_reio': sample.get('tau_reio', LCDM.tau_reio),
        'm_ncdm': sample.get('m_ncdm', LCDM.m_ncdm) / (sample.get('N_ncdm', 1) * sample.get('deg_ncdm', 1)),
        'sigma_m': sample.get('m_ncdm', LCDM.m_ncdm),
        'N_eff': sample.get('N_eff', LCDM.Neff),
        'r': sample.get('r', 0.05),
        'log10T_heat_hmcode': sample.get('log10T_heat_hmcode'),
        'w0_fld': sample.get('w0_fld', -1.0),
        'wa_fld': sample.get('wa_fld', 0.0),
        'Omega_k': sample.get('Omega_k', 0.0),
        'N_ncdm': sample.get('N_ncdm', 1),
        'z_pk': sample.get('z_pk', 2.5),
        'N_ur': sample.get('N_ur', max(LCDM.Neff - 3, 0.0)),
        'deg_ncdm': sample.get('deg_ncdm', 3),
        'T_ncdm': sample.get('T_ncdm', 0.71611 / ((4.0/11.0)**(1.0/3.0))),
    }
    full['A_s'] = logA_to_As(full['logA'])
    return full


def build_camb_class_input(full_params: Dict[str, float]) -> Dict[str, float]:
    """Translate parameter dictionary into the structure expected by CAMB/CLASS."""
    return {
        'H0': float(full_params['H0']),
        'ombh2': float(full_params['omega_b']),
        'omch2': float(full_params['omega_cdm']),
        'tau': float(full_params['tau_reio']),
        'ns': float(full_params['n_s']),
        'As': float(full_params['A_s']),
        'N_eff': float(full_params['N_eff']),
        'sigma_m': float(full_params['sigma_m']),
        'm_ncdm': float(full_params['m_ncdm']),
        'r': float(full_params['r']),
        'log10T_heat_hmcode': float(full_params['log10T_heat_hmcode']),
        'w0_fld': float(full_params['w0_fld']),
        'wa_fld': float(full_params['wa_fld']),
        'Omega_k': float(full_params['Omega_k']),
        'z_pk': float(full_params['z_pk']),
        'N_ncdm': int(full_params['N_ncdm']),
        'N_ur': full_params['N_ur'],
        'T_ncdm': full_params['T_ncdm'],
        'deg_ncdm': int(full_params['deg_ncdm']),
        'm_ncdm_str': ','.join([f'{full_params["m_ncdm"]:.32f}'] * int(full_params["deg_ncdm"])),
        'T_ncdm_str': ','.join([f'{full_params["T_ncdm"]:.32f}'] * int(full_params["deg_ncdm"])),
    }


# --------------------------------------------------------------------------------------
# CAMB / CLASS configuration (copied + lightly parameterised from Cl_CLASS_CAMB_Test.py)
# --------------------------------------------------------------------------------------


def configure_camb_params(p: Dict[str, float], nonlinear: bool = True, lmax: int = L_MAX, BB: bool = False) -> CAMBparams:
    params = CAMBparams()

    params.set_cosmology(
        H0=p['H0'],
        ombh2=p['ombh2'],
        omch2=p['omch2'],
        tau=p['tau'],
        mnu=p.get('m_ncdm', LCDM.m_ncdm)*p['deg_ncdm'],
        num_massive_neutrinos=p.get('deg_ncdm', 1),
        nnu=p.get('N_eff', LCDM.Neff),
        omk=0.0,
    )

    if BB:
        params.WantTensors = True
        params.InitPower.set_params(
            ns=p['ns'],
            As=p['As'],
            r=p.get('r', 0.05),
            nt=None,
            pivot_scalar=0.05,
            pivot_tensor=0.05,
        )
    else:
        params.InitPower.set_params(ns=p['ns'], As=p['As'])

    params.set_accuracy(lAccuracyBoost=2.2, AccuracyBoost=2.0, lSampleBoost=2.0)
    params.set_for_lmax(lmax=lmax, lens_potential_accuracy=15, lens_margin=2050)

    acc = params.Accuracy
    acc.min_l_logl_sampling = 6000

    if BB:
        params.max_l_tensor = lmax
        acc = params.Accuracy
        acc.k_eta_max_tensor = max(getattr(acc, 'k_eta_max_tensor', 0.0), 3.5 * lmax)
        acc.k_eta_max_scalar = max(getattr(acc, 'k_eta_max_scalar', 3.0 * lmax), 3.0 * lmax)

    params.NonLinear = model.NonLinear_both if nonlinear else model.NonLinear_none

    params.set_matter_power(redshifts=[p.get('z_pk')], kmax=10, k_per_logint=130)

    params.AccurateMassiveNeutrinoTransfers = True
    params.DoLateRadTruncation = False
    params.recombination_model = "HyRec"
    params.TCMB = Planck.Tcmb0.value
    params.NonLinearModel.set_params(
        halofit_version='mead2020_feedback',
        HMCode_logT_AGN=p.get('log10T_heat_hmcode')
    )

    params.set_dark_energy(w=p.get('w0_fld', -1.0), wa=p.get('wa_fld', 0.0))
    
    return params


def configure_class(p: Dict[str, float], lmax: int = L_MAX, nonlinear: bool = True, BB: bool = False) -> Dict[str, float]:
    # --- Precompute density parameters
    h = p['H0'] / 100.0
    Omega_b = p['ombh2'] / h**2
    Omega_cdm = p['omch2'] / h**2
    Omega_k = 0.0
    Omega_fld = 1.0 - (Omega_b + Omega_cdm + Omega_k)

    class_params = {
        # Outputs
        'output': 'lCl,tCl,pCl,mPk',
        'modes': 's',

        # Cosmology
        'h': p['H0'] / 100.0,
        'Omega_b': p['ombh2'] / (p['H0'] / 100.0) ** 2,
        'Omega_cdm': p['omch2'] / (p['H0'] / 100.0) ** 2,
        'Omega_k': 0.0,
        'tau_reio': p['tau'],
        'A_s': p['As'],
        'n_s': p['ns'],
        'T_cmb': Planck.Tcmb0.value,
        'YHe': 'BBN',
        'z_max_pk': 35,
        'N_ncdm': int(p.get('deg_ncdm', 1)),
        'm_ncdm': p['m_ncdm_str'],
        'N_ur': p['N_ur'],
        'T_ncdm': p['T_ncdm_str'],

        # Accuracy controls
        'non_linear': 'hmcode',
        'l_max_scalars': lmax,
        'hmcode_version': '2020_baryonic_feedback',
        'P_k_max_h/Mpc': 100.0,
        'delta_l_max': 1800,
        'l_logstep': 1.025,
        'l_linstep': 20,
        'perturbations_sampling_stepsize': 0.05,
        'l_switch_limber': 30.0,
        'hyper_sampling_flat': 32.0,
        'l_max_g': 40,
        'l_max_ur': 35,
        'l_max_pol_g': 60,
        'l_max_ncdm': 30,
        'ur_fluid_approximation': 2,
        'ur_fluid_trigger_tau_over_tau_k': 130.0,
        'radiation_streaming_approximation': 2,
        'radiation_streaming_trigger_tau_over_tau_k': 240.0,
        'hyper_flat_approximation_nu': 7000.0,
        'transfer_neglect_delta_k_S_t0': 0.17,
        'transfer_neglect_delta_k_S_t1': 0.05,
        'transfer_neglect_delta_k_S_t2': 0.17,
        'transfer_neglect_delta_k_S_e': 0.17,
        'accurate_lensing': 1,
        'start_small_k_at_tau_c_over_tau_h': 0.0004,
        'start_large_k_at_tau_h_over_tau_k': 0.05,
        'tight_coupling_trigger_tau_c_over_tau_h': 0.005,
        'tight_coupling_trigger_tau_c_over_tau_k': 0.008,
        'start_sources_at_tau_c_over_tau_h': 0.006,
        'tol_ncdm_synchronous': 1e-6,
        'recombination': 'HyRec',
        'sBBN file': '/external/bbn/sBBN_2021.dat',
        'lensing': 'yes',
        'log10T_heat_hmcode': p.get('log10T_heat_hmcode'),
        'fluid_equation_of_state': 'CLP',
        'Omega_fld': Omega_fld,
        'w0_fld': p.get('w0_fld', -1.0),
        'wa_fld': p.get('wa_fld', 0.0),
    }

    if BB:
        class_params.update({
            'modes': 's,t',
            'r': p.get('r', 0.05),
            'k_pivot': 0.05,
            'reio_parametrization': 'reio_camb',
            'l_max_tensors': lmax,
        })

    return class_params


def quick_validate_cosmology(p: Dict[str, float]) -> Tuple[bool, str]:
    """
    Fast, low-accuracy validation for a cosmology.
    Ensures both CLASS and CAMB can initialize and compute a minimal set of quantities.
    """

    # --- CAMB test (fast)
    try:
        print("Trying CAMB")
        camb_params = configure_camb_params(p, nonlinear=False, lmax=200)
        camb_params.set_accuracy(AccuracyBoost=0.5, lAccuracyBoost=0.5, lSampleBoost=0.5)
        camb_params.set_for_lmax(lmax=200, lens_potential_accuracy=1)
        camb_params.set_matter_power(redshifts=[0.0], kmax=35.0)
        results = camb.get_results(camb_params)
        _ = results.get_sigma8()
    except Exception as e:
        return False, f"CAMB failed: {repr(e)}"

    print("CAMB passed")

    # --- CLASS test (fast)
    try:
        print("Trying CLASS")
        cl = classy.Class()
        test_params = configure_class(p, lmax=200, nonlinear=False, BB=False)
        cl.set(test_params)
        cl.compute()
        pk_val = cl.pk(0.1, 0.0)
        if not np.isfinite(pk_val) or pk_val <= 0:
            return False, f"CLASS invalid pk={pk_val}"
        cl.struct_cleanup()
        cl.empty()
    except Exception as e:
        return False, f"CLASS failed: {repr(e)}"

    print("CLASS passed")

    return True, "Both CLASS and CAMB validated ✅"


# --------------------------------------------------------------------------------------
# Observable computation helpers
# --------------------------------------------------------------------------------------


def compute_class_observables(p: Dict[str, float]) -> Dict[str, np.ndarray]:
    print("  [CLASS] Setting nonlinear CLASS parameters...")
    class_params_nl = configure_class(p, lmax=L_MAX, nonlinear=True, BB=False)
    cl_nl = classy.Class()

    print("  [CLASS] Computing nonlinear CLASS model...")
    cl_nl.set(class_params_nl)
    cl_nl.compute()
    print("  [CLASS] Nonlinear CLASS computation finished ✅")

    print("  [CLASS] Extracting lensed/unlensed Cls...")
    cls_l = cl_nl.lensed_cl(lmax=L_MAX)
    ell_lensed = cls_l['ell']
    ell_factor = ell_lensed * (ell_lensed + 1) / (2 * np.pi)

    cls_raw = cl_nl.raw_cl(lmax=L_MAX)
    ell_unlensed = cls_raw['ell']
    ell_factor_unlensed = ell_unlensed * (ell_unlensed + 1) / (2 * np.pi)

    print("  [CLASS] Computing nonlinear matter power spectra (P(k))...")
    pk_nonlin = np.array([cl_nl.pk(k, p.get('z_pk')) for k in K_VALS])
    pk_nonlin_cb = np.array([cl_nl.pk_cb(k, p.get('z_pk')) for k in K_VALS])

    print("  [CLASS] Computing background quantities (H(z), D_A(z))...")
    H_class = np.array([cl_nl.Hubble(z) for z in Z_BG]) * 299792.458
    DA_class = np.array([cl_nl.angular_distance(z) for z in Z_BG])

    print("  [CLASS] Extracting derived parameters...")
    derived = cl_nl.get_current_derived_parameters(
        ['sigma8', 'z_reio', 'z_d', 'rs_d', 'z_rec', 'rs_rec', '100*theta_s', 'YHe']
    )

    print("  [CLASS] Cleaning up nonlinear CLASS instance...")
    cl_nl.struct_cleanup()
    cl_nl.empty()

    print("  [CLASS] Running linear calculation...")
    class_params_lin = configure_class(p, lmax=L_MAX, nonlinear=False, BB=False)
    cl_lin = classy.Class()
    cl_lin.set(class_params_lin)
    cl_lin.compute()

    print("  [CLASS] Computing nonlinear matter power spectra (P(k))...")
    pk_lin = np.array([cl_lin.pk_lin(k, p.get('z_pk')) for k in K_VALS])
    pk_lin_cb = np.array([cl_lin.pk_cb_lin(k, p.get('z_pk')) for k in K_VALS])

    print("  [CLASS] Computing scale-independent growth quantities...")
    k0 = 1e-5  # small k for scale independent limit
    growth_factor = np.array([cl_lin.scale_dependent_growth_factor_D(k0, z) for z in Z_BG])
    growth_factor /= growth_factor[0]  # normalize
    lnD = np.log(growth_factor)
    dlnD_dz = np.gradient(lnD, Z_BG)
    growth_rate = -(1 + Z_BG) * dlnD_dz

    print("  [CLASS] Cleaning up linear CLASS instance...")
    cl_lin.struct_cleanup()
    cl_lin.empty()

    print("  [CLASS] Running B-mode (BB) tensor calculation...")
    class_params_bb = configure_class(p, lmax=L_MAX, nonlinear=True, BB=True)
    cl_bb = classy.Class()
    cl_bb.set(class_params_bb)
    cl_bb.compute()

    cls_bb = cl_bb.raw_cl(lmax=L_MAX)
    ell_unlensed_bb = cls_bb['ell']
    ell_factor_bb = ell_unlensed_bb * (ell_unlensed_bb + 1) / (2 * np.pi)
    cl_bb_unlensed = cls_bb['bb'] * ell_factor_bb
    print("  [CLASS] B-mode computation complete ✅")

    cl_bb.struct_cleanup()
    cl_bb.empty()
    print("  [CLASS] Finished full CLASS observable computation ✅")

    return {
        'ell_lensed': ell_lensed[2:],
        'cl_tt_lensed': cls_l['tt'][2:] * ell_factor[2:],
        'cl_ee_lensed': cls_l['ee'][2:] * ell_factor[2:],
        'cl_bb_lensed': cls_l['bb'][2:] * ell_factor[2:],
        'cl_te_lensed': cls_l['te'][2:] * ell_factor[2:],
        'ell_phi_phi': ell_lensed[2:],
        'cl_phi_phi': cls_l['pp'][2:],
        'ell_unlensed': ell_unlensed[2:],
        'cl_tt_unlensed': cls_raw['tt'][2:] * ell_factor_unlensed[2:],
        'cl_ee_unlensed': cls_raw['ee'][2:] * ell_factor_unlensed[2:],
        'cl_te_unlensed': cls_raw['te'][2:] * ell_factor_unlensed[2:],
        'ell_bb_unlensed': np.arange(2, 2 + len(cl_bb_unlensed[2:501])),
        'cl_bb_unlensed': cl_bb_unlensed[2:501],
        'pk_k': K_VALS,
        'pk_linear': pk_lin,
        'pk_nonlinear': pk_nonlin,
        'pk_cb_linear': pk_lin_cb,
        'pk_cb_nonlinear': pk_nonlin_cb,
        'background_z': Z_BG,
        'background_Hz': H_class,
        'background_DAz': DA_class,
        'scale_independent_growth_factor': growth_factor,
        'scale_independent_growth_rate': growth_rate,
        'derived_theta_star': derived['100*theta_s'],
        'derived_sigma8': derived['sigma8'],
        'derived_YHe': derived['YHe'],
        'derived_z_reio': derived['z_reio'],
        'derived_z_star': derived['z_rec'],
        'derived_r_star': derived['rs_rec'],
        'derived_z_drag': derived['z_d'],
        'derived_r_drag': derived['rs_d'],
    }


def compute_camb_observables(p: Dict[str, float]) -> Dict[str, np.ndarray]:
    print("  [CAMB] Setting nonlinear CAMB parameters...")
    camb_params_nl = configure_camb_params(p, nonlinear=True, lmax=L_MAX, BB=False)

    print("  [CAMB] Computing nonlinear CAMB model...")
    camb_results_nl = camb.get_results(camb_params_nl)
    print("  [CAMB] Nonlinear CAMB computation finished ✅")

    print("  [CAMB] Extracting lensed and unlensed scalar Cls...")
    lensed_cls = camb_results_nl.get_lensed_scalar_cls(lmax=L_MAX)
    unlensed_cls = camb_results_nl.get_unlensed_scalar_cls(lmax=L_MAX)
    lp_camb = camb_results_nl.get_lens_potential_cls(lmax=L_MAX, raw_cl=True)
    print("  [CAMB] Cl extraction complete ✅")

    print("  [CAMB] Computing nonlinear matter power spectra (P(k))...")
    pk_nonlin = camb.get_matter_power_interpolator(
        camb_params_nl, nonlinear=True, hubble_units=False, k_hunit=False, kmax=35, zmax=3
    ).P(p.get('z_pk'), K_VALS)
    print("  [CAMB] Nonlinear P(k) computed ✅")

    print("  [CAMB] Computing nonlinear CDM+baryon matter power spectra (P_cb(k))...")
    pk_nonlin_cb = camb.get_matter_power_interpolator(
        camb_params_nl, nonlinear=True, hubble_units=False, k_hunit=False, kmax=35, zmax=3,
        var1='delta_nonu', var2='delta_nonu'
    ).P(p.get('z_pk'), K_VALS)
    print("  [CAMB] Nonlinear P_cb(k) computed ✅")

    print("  [CAMB] Computing background quantities (H(z), D_A(z))...")
    H_camb = np.array([camb_results_nl.hubble_parameter(z) for z in Z_BG])
    DA_camb = np.array([camb_results_nl.angular_diameter_distance(z) for z in Z_BG])
    print("  [CAMB] Background quantities computed ✅")

    print("  [CAMB] Extracting derived parameters (θ*, σ₈, YHe, etc.)...")
    param_dict_z0 = dict(p)
    param_dict_z0['z_pk'] = 0.0
    camb_params_nl_zero = configure_camb_params(param_dict_z0, nonlinear=True, lmax=L_MAX)
    camb_results_nl_zero = camb.get_results(camb_params_nl_zero)

    derived = camb_results_nl_zero.get_derived_params()
    derived_dict = {
        'theta_star': derived['thetastar'],
        'sigma8': camb_results_nl_zero.get_sigma8_0(),
        'YHe': camb_params_nl_zero.YHe,
        'z_reio': camb_params_nl_zero.Reion.get_zre(camb_params_nl_zero),
        'z_star': derived['zstar'],
        'r_star': derived['rstar'],
        'z_drag': derived['zdrag'],
        'r_drag': derived['rdrag'],
    }
    print("  [CAMB] Derived parameters extracted ✅")

    print("  [CAMB] Computing linear matter power spectra and growth quantities...")
    camb_params_lin = configure_camb_params(p, nonlinear=False, lmax=L_MAX)
    growth_redshifts = sorted({
        0.0,
        float(p.get('z_pk', 0.0)),
        float(np.max(Z_BG))
    })
    camb_params_lin.set_matter_power(
        redshifts=growth_redshifts,
        kmax=10,
        k_per_logint=130
    )
    zmax_growth = max(float(np.max(Z_BG)), float(p.get('z_pk', 0.0)))
    pk_interp_lin = camb.get_matter_power_interpolator(
        camb_params_lin, nonlinear=False, hubble_units=False, k_hunit=False, kmax=35, zmax=zmax_growth
    )
    pk_lin = pk_interp_lin.P(p.get('z_pk'), K_VALS)
    growth_k = 1e-5
    growth_pk = np.array([pk_interp_lin.P(z, growth_k) for z in Z_BG])
    growth_pk0 = np.maximum(growth_pk[0], 1e-30)
    growth_factor = np.sqrt(np.maximum(growth_pk, 1e-30) / growth_pk0)
    growth_factor = growth_factor / growth_factor[0]
    lnD = np.log(growth_factor)
    dlnD_dz = np.gradient(lnD, Z_BG)
    growth_rate = -(1 + Z_BG) * dlnD_dz

    print("  [CAMB] Linear P(k) and growth quantities computed ✅")

    print("  [CAMB] Computing linear CDM+baryon matter power spectra (P_cb_lin(k))...")
    pk_interp_lin_cb = camb.get_matter_power_interpolator(
        camb_params_lin, nonlinear=False, hubble_units=False, k_hunit=False, kmax=35, zmax=zmax_growth,
        var1='delta_nonu', var2='delta_nonu'
    )
    pk_lin_cb = pk_interp_lin_cb.P(p.get('z_pk'), K_VALS)
    print("  [CAMB] Linear P_cb(k) computed ✅")

    print("  [CAMB] Running B-mode (BB) tensor calculation...")
    camb_params_bb = configure_camb_params(p, nonlinear=True, lmax=L_MAX, BB=True)
    camb_results_bb = camb.get_results(camb_params_bb)
    bb_unlensed = camb_results_bb.get_unlensed_total_cls(lmax=L_MAX)
    print("  [CAMB] B-mode (BB) computation complete ✅")

    print("  [CAMB] Finished full CAMB observable computation ✅")

    return {
        'ell_lensed': np.arange(lensed_cls.shape[0])[2:],
        'cl_tt_lensed': lensed_cls.T[0][2:],
        'cl_ee_lensed': lensed_cls.T[1][2:],
        'cl_bb_lensed': lensed_cls.T[2][2:],
        'cl_te_lensed': lensed_cls.T[3][2:],
        'ell_phi_phi': np.arange(lp_camb.shape[0])[2:],
        'cl_phi_phi': lp_camb[2:, 0],
        'ell_unlensed': np.arange(unlensed_cls.shape[0])[2:],
        'cl_tt_unlensed': unlensed_cls.T[0][2:],
        'cl_ee_unlensed': unlensed_cls.T[1][2:],
        'cl_te_unlensed': unlensed_cls.T[3][2:],
        'ell_bb_unlensed': np.arange(2, 2 + len(bb_unlensed.T[2][2:501])),
        'cl_bb_unlensed': bb_unlensed.T[2][2:501],
        'pk_k': K_VALS,
        'pk_nonlinear': pk_nonlin,
        'pk_linear': pk_lin,
        'pk_cb_nonlinear': pk_nonlin_cb,
        'pk_cb_linear': pk_lin_cb,
        'background_z': Z_BG,
        'background_Hz': H_camb,
        'background_DAz': DA_camb,
        'scale_independent_growth_factor': growth_factor,
        'scale_independent_growth_rate': growth_rate,
        'derived_theta_star': derived_dict['theta_star'],
        'derived_sigma8': derived_dict['sigma8'],
        'derived_YHe': derived_dict['YHe'],
        'derived_z_reio': derived_dict['z_reio'],
        'derived_z_star': derived_dict['z_star'],
        'derived_r_star': derived_dict['r_star'],
        'derived_z_drag': derived_dict['z_drag'],
        'derived_r_drag': derived_dict['r_drag'],
    }


# --------------------------------------------------------------------------------------
# Persistence helpers
# --------------------------------------------------------------------------------------


def save_npz(path: Path, data: Dict[str, np.ndarray]) -> None:
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in data.items()}
    np.savez_compressed(path, **arrays)


def save_parameters(path: Path, params: Dict[str, float]) -> None:
    arrays = {key: np.array(value, dtype=np.float32) for key, value in params.items()}
    np.savez_compressed(path, **arrays)


def append_error_log(path: Path, entry: Dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------------------
# Main execution loop
# --------------------------------------------------------------------------------------


def run_single_batch(config: SamplerConfig, batch_idx: int) -> None:
    yaml_config = load_yaml_config(config.yaml_file)
    param_specs = extract_param_specs(yaml_config)
    print(f"Extracted parameter specs: {param_specs}")

    rng = np.random.default_rng(config.seed + batch_idx if config.seed is not None else None)
    pool = LHSSamplePool(param_specs, rng=rng, chunk_size=config.chunk_size)
    print(f"Sampling from {len(param_specs)} parameters: {list(param_specs.keys())}")

    batch_dir = config.output_dir / f"batch_{batch_idx:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    error_log_path = config.output_dir / ERROR_LOG_FILENAME
    if error_log_path.exists() and batch_idx == 1:
        error_log_path.unlink()

    set_counter = 0
    camb_times, class_times = [], []

    while set_counter < config.batch_size:
        sample = pool.next()
        full_params = build_full_parameter_dict(sample)
        params32 = {k: np.float32(v) if isinstance(v, (float, int)) else v for k, v in full_params.items()}
        camb_class_input = build_camb_class_input(full_params)
        print(f"[Batch {batch_idx}, Set {set_counter+1}] Candidate params: {full_params}")

        # --- 🔍 Quick validation before heavy computation
        valid, msg = quick_validate_cosmology(camb_class_input)
        if not valid:
            print(f"[Batch {batch_idx}, Set {set_counter+1}] Validation failed: {msg}. Retrying...")
            continue
        else:
            print(f"[Batch {batch_idx}, Set {set_counter+1}] Validation passed ✅")

        # --- CAMB computation (safe subprocess)
        t0 = time.perf_counter()
        camb_data, camb_error = safe_compute(compute_camb_observables, (camb_class_input,), timeout=2300)
        if camb_data is None:
            append_error_log(error_log_path, {
                'batch': batch_idx,
                'set_index': set_counter + 1,
                'stage': 'CAMB',
                'parameters': full_params,
                'error': camb_error,
            })
            print(f"[Batch {batch_idx}, Set {set_counter+1}] CAMB failed: {camb_error}")
            continue
        camb_time = time.perf_counter() - t0
        camb_times.append(camb_time)
        print(f"[Batch {batch_idx}, Set {set_counter+1}] CAMB complete in {camb_time:.2f} s ✅")

        # --- CLASS computation (safe subprocess)
        t1 = time.perf_counter()
        class_data, class_error = safe_compute(compute_class_observables, (camb_class_input,), timeout=2000)
        if class_data is None:
            append_error_log(error_log_path, {
                'batch': batch_idx,
                'set_index': set_counter + 1,
                'stage': 'CLASS',
                'parameters': full_params,
                'error': class_error,
            })
            print(f"[Batch {batch_idx}, Set {set_counter+1}] CLASS failed: {class_error}")
            continue
        class_time = time.perf_counter() - t1
        class_times.append(class_time)
        print(f"[Batch {batch_idx}, Set {set_counter+1}] CLASS complete in {class_time:.2f} s ✅")

        # --- Save results
        set_idx = set_counter + 1
        save_parameters(batch_dir / f"set_{set_idx:02d}_params", params32)
        save_npz(batch_dir / f"set_{set_idx:02d}_class", class_data)
        save_npz(batch_dir / f"set_{set_idx:02d}_camb", camb_data)
        print(f"[Batch {batch_idx}, Set {set_idx}] Saved successfully.")
        set_counter += 1
        
    # --- Compute and print timing statistics
    if camb_times:
        camb_avg = np.mean(camb_times)
        camb_std = np.std(camb_times)
        print(f"\n[Batch {batch_idx}] ⏱️ CAMB timings: avg = {camb_avg:.2f} s ± {camb_std:.2f} s over {len(camb_times)} runs")
    else:
        print(f"\n[Batch {batch_idx}] ⚠️ No successful CAMB runs to compute timing stats")

    if class_times:
        class_avg = np.mean(class_times)
        class_std = np.std(class_times)
        print(f"[Batch {batch_idx}] ⏱️ CLASS timings: avg = {class_avg:.2f} s ± {class_std:.2f} s over {len(class_times)} runs\n")
    else:
        print(f"[Batch {batch_idx}] ⚠️ No successful CLASS runs to compute timing stats\n")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args() -> SamplerConfig:
    parser = argparse.ArgumentParser(description="Batch CLASS/CAMB data generator")
    parser.add_argument(
        '--yaml-file',
        type=Path,
        default='/home/jam249/rds/rds-dirac-dp002/jam249/Neural-Net-Emulator-for-Cosmological-Observables/Data-Generation/Parallel-Data-Gen/parameter-ranges-test.yaml',
        help='Cobaya YAML file used to define cosmological parameter priors.'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default='/home/jam249/rds/rds-dirac-dp002/jam249/Neural-Net-Emulator-for-Cosmological-Observables/Data-Generation/Parallel-Data-Gen/Slurm_Batch/CAMB-CLASS/sh_batch_outputs',
        help='Directory where batch outputs will be written.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Number of parameter sets per batch (default: 2).'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Optional random seed for reproducibility.'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=256,
        help='Number of LHS samples to draw whenever the pool is replenished.'
    )
    parser.add_argument(
        '--batch-idx',
        type=int,
        required=False,
        default=1,
        help='Index of this SLURM batch (1-based).'
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SamplerConfig(
        yaml_file=args.yaml_file.resolve(),
        n_batches=1,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    run_single_batch(config, args.batch_idx)


if __name__ == '__main__':
    t_0 = time.perf_counter()
    main()
    t_1 = time.perf_counter()
    print(f"Total was: {t_1-t_0}s")
