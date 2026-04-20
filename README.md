# Emulating-Cosmological-Observables-with-Neural-Networks

This project develops my Master's project to emulate more cosmological observables to higher quality.

This will contain a set of neural networks (MLP architecture) in JAX for rapid emulation of comsological observables for use in cosmological parameter inference. The observables from CAMB and CLASS are saved in float32 in .npz files.

We use precision parameters from [this dr6 paper](https://arxiv.org/pdf/2503.14454) appendix A. Please check carefully. 

The observables that are emulated are:

# Emulated Observables
- Linear matter power spectrum
- Nonlinear matter power spectrum
- Linear matter power spectrum (CB only)
- Nonlinear matter power spectrum (CB only)
- Cl_ee lensed spectrum
- Cl_ee unlensed spectrum
- Cl_tt lensed spectrum
- Cl_tt unlensed spectrum
- Cl_te lensed spectrum
- Cl_te unlensed spectrum
- Cl_bb lensed spectrum
- Cl_bb unlensed spectrum
- Cl_phi_phi
- D_A(z)
- H(z)
- Derived paramteres:
    - simga8(z=0)
    - theta*
    - YHe
    - z_reio
    - z_star
    - r_star
    - z_drag
    - r_drag
 
# Cosmological Parameter Emulation Ranges

| **Parameter**        | **Lower** | **Upper** | **Description**                                                            |
| -------------------- | --------- | --------- | -------------------------------------------------------------------------- |
| `H0`                 | 20        | 100       | Hubble constant *(km s⁻¹ Mpc⁻¹)*                                           |
| `omega_b`            | 0.001     | 0.030     | Physical baryon density, Ω<sub>b</sub>h²                                   |
| `omega_cdm`          | 0.001     | 0.25      | Physical cold dark matter density, Ω<sub>cdm</sub>h²                       |
| `n_s`                | 0.8       | 1.2       | Scalar spectral index                                                      |
| `tau_reio`           | 0.01      | 0.3       | Optical depth to reionization                                              |
| `logA`               | 1.61      | 3.91      | log<sub>10</sub>(10¹⁰ A<sub>s</sub>), amplitude of primordial fluctuations |
| `N_eff`              | 0         | 7.5       | Effective number of relativistic species                                   |
| `m_ncdm`             | 0         | 5         | Total neutrino mass Σ m<sub>ν</sub> (eV)                                   |
| `log10T_heat_hmcode` | 7.0       | 8.5       | Logarithm of HMCode heating temperature                                    |
| `w0_fld`             | −3        | 1         | Dark energy equation-of-state parameter w₀                                 |
| `wa_fld`             | −3        | 2         | Dark energy evolution parameter wₐ                                         |
| `Omega_k`            | −0.8      | 0.8       | Curvature parameter Ω<sub>k</sub>                                          |
| `A_b`                | 2         | 4         | HMCode feedback amplitude                                                  |
| `eta_b`              | 0.5       | 1.0       | HMCode feedback slope                                                      |


# CAMB Accuracy Settings
## Accurcy controls
- AccuracyBoost: 2.0
- DoLateRadTruncation: false
- TCMB: 2.7255
- accurate_massve_neutrino _transfer: true
- halofit_version: mead2020
- k_per_logint: 130
- kmax: 10
- lAccuracyBoost: 2.2
- lSampleBoost: 2.0
- lens_margin: 2050
- lens_potential_accuracy: 15
- lmax: 10,000
- min_l_logl_sampling: 6000 - need to add in
- nonlinear: true
- recombination_model: HyRec

## BB unlensed accuracy controls:
- max_l_tensor: 10,000
- k_eta_max_tensor: max(getattr(acc, ‘k_eta_max_tensor’, 3.0*lmax), 3.0*lmax)
- k_eta_max_scalar: max(getattr(acc, ‘k_eta_max_scalar’, 3.0*lmax), 3.0*lmax)
- pivot_scalar: 0.05
- pivot_tensor: 0.05

# CLASS Accuracy Settings
## Outputs
- 'output': 'lCl,tCl,pCl,mPk
- 'modes': 's'
- 
## Accuracy controls
- 'P_k_max_h/Mpc': 100.0
- 'l_max_scalars': 10,000
- 'delta_l_max': 1800
- 'l_logstep': 1.025
- 'l_linstep': 20
- 'perturbations_sampling_stepsize': 0.05
- 'l_switch_limber': 30.0
- 'hyper_sampling_flat': 32.0
- 'l_max_g': 40
- 'l_max_ur': 35
- 'l_max_pol_g': 60
- 'l_max_ncdm': 30
- 'ur_fluid_approximation': 2
- 'ur_fluid_trigger_tau_over_tau_k': 130.0
- 'radiation_streaming_approximation': 2
- 'radiation_streaming_trigger_tau_over_tau_k': 240.0
- 'hyper_flat_approximation_nu': 7000.0
- 'transfer_neglect_delta_k_S_t0': 0.17
- 'transfer_neglect_delta_k_S_t1': 0.05
- 'transfer_neglect_delta_k_S_t2': 0.17
- 'transfer_neglect_delta_k_S_e': 0.17
- 'accurate_lensing': 1
- 'start_small_k_at_tau_c_over_tau_h': 0.0004
- 'start_large_k_at_tau_h_over_tau_k': 0.05
- 'tight_coupling_trigger_tau_c_over_tau_h': 0.005
- ‘tight_coupling_trigger_tau_c_over_tau_k': 0.008
- 'start_sources_at_tau_c_over_tau_h': 0.006
- 'tol_ncdm_synchronous': 1e-6
- 'recombination': 'HyRec'
- 'sBBN file': '/external/bbn/sBBN_2021.dat'
- 'non_linear': 'hmcode'
- 'lensing': 'yes'
- 'N_ur': (N_eff - 3.03959) and 'T_ncdm' : 0.71611  if N_eff > 3.03959
- 'N_ur' : 0  and 'T_ncdm' : 0.71611*(N_eff / 3.03959)**0.25 if N_eff < 3.03959

## BB unlensed accuracy controls:
- ‘modes’: ‘s, t’
- ‘k_pivot’: 0.05
- ‘reio_parametrization’: ‘reio_camb’
- ‘l_max_tensors’: 10,000

