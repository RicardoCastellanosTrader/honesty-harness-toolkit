#!/usr/bin/env python3
"""
lab_cuda.py - Port CUDA de run_on_slice del motor de backtesting.

Ejecuta la simulacion de trading en GPU usando Numba CUDA.
Produce resultados identicos (dentro de tolerancia float64) al motor CPU
de lab_historico_numba_v8_3.py.

Arquitectura:
  - precalculate_all_data se queda en CPU (se ejecuta pocas veces)
  - Los datos precalculados se copian a GPU una vez (upload_data)
  - Cada CUDA thread simula 1 config (paralelismo perfecto)
  - calc_score_v63 se queda en CPU (se ejecuta sobre resultados en host)

Uso:
    from lab_cuda import CUDASimulator
    sim = CUDASimulator()
    handle = sim.upload_data(data_dict)
    results = sim.run_on_slice(configs, handle, start, end, sl, sl_e, ts, cd, comm)
"""

import numpy as np
import math
import os
import sys
import ctypes

# ============================================
# CUDA TOOLKIT SETUP (must run before importing numba.cuda)
# ============================================
def _setup_cuda_paths():
    """Configure CUDA toolkit paths for Numba on Windows.

    Handles compatibility issues between CUDA toolkit, driver, and Numba:
    - Prefers pip-installed CUDA 12.x packages (nvidia-cuda-nvcc-cu12,
      nvidia-cuda-runtime-cu12) which match the driver's CUDA 12.9 support.
    - Falls back to system CUDA toolkit if pip packages not available.
    - Adds CC 12.0 (Blackwell/RTX 50xx) to Numba's support tables.
    """
    if sys.platform != 'win32':
        return

    import glob
    import site

    # --- Find DLLs: prefer pip packages (CUDA 12.x), fallback to system ---
    site_packages = None
    all_site_dirs = list(site.getsitepackages())
    try:
        all_site_dirs.append(site.getusersitepackages())
    except AttributeError:
        pass
    for sp in all_site_dirs:
        nvidia_dir = os.path.join(sp, 'nvidia')
        if os.path.isdir(nvidia_dir):
            site_packages = sp
            break

    nvvm_dll_path = None
    cudart_dll_path = None
    libdevice_path = None

    # Try pip packages first
    if site_packages:
        nvidia_dir = os.path.join(site_packages, 'nvidia')
        # NVVM from nvidia-cuda-nvcc-cu12
        nvvm_candidates = glob.glob(os.path.join(nvidia_dir, 'cuda_nvcc', 'nvvm', 'bin', 'nvvm64_*.dll'))
        if nvvm_candidates:
            nvvm_dll_path = max(nvvm_candidates)
        # libdevice from same package
        ld = os.path.join(nvidia_dir, 'cuda_nvcc', 'nvvm', 'libdevice', 'libdevice.10.bc')
        if os.path.exists(ld):
            libdevice_path = ld
        # cudart from nvidia-cuda-runtime-cu12
        cudart_candidates = glob.glob(os.path.join(nvidia_dir, 'cuda_runtime', 'bin', 'cudart64_*.dll'))
        if cudart_candidates:
            cudart_dll_path = max(cudart_candidates)

    # Fallback to system CUDA toolkit
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if not cuda_home:
        for base in [r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA']:
            if os.path.isdir(base):
                versions = sorted(os.listdir(base), reverse=True)
                if versions:
                    cuda_home = os.path.join(base, versions[0])
                    os.environ['CUDA_HOME'] = cuda_home
                    break

    if cuda_home and os.path.isdir(cuda_home):
        if not nvvm_dll_path:
            candidates = glob.glob(os.path.join(cuda_home, 'nvvm', 'bin', 'x64', 'nvvm64_*.dll'))
            if candidates:
                nvvm_dll_path = max(candidates)
        if not libdevice_path:
            ld = os.path.join(cuda_home, 'nvvm', 'libdevice', 'libdevice.10.bc')
            if os.path.exists(ld):
                libdevice_path = ld
        if not cudart_dll_path:
            for subdir in ['bin/x64', 'bin']:
                candidates = glob.glob(os.path.join(cuda_home, subdir, 'cudart64_*.dll'))
                if candidates:
                    cudart_dll_path = max(candidates)
                    break

        # Add DLL directories
        for subdir in ['nvvm/bin/x64', 'bin', 'bin/x64']:
            d = os.path.join(cuda_home, subdir)
            if os.path.isdir(d):
                try:
                    os.add_dll_directory(d)
                except (OSError, AttributeError):
                    pass

    # Add DLL directories for pip package paths
    for dll_path in [nvvm_dll_path, cudart_dll_path]:
        if dll_path:
            d = os.path.dirname(dll_path)
            try:
                os.add_dll_directory(d)
            except (OSError, AttributeError):
                pass

    # --- Patch Numba's library resolution ---
    try:
        import numba.cuda.cudadrv.libs as _cuda_libs

        if nvvm_dll_path:
            _original_get_cudalib = _cuda_libs.get_cudalib
            _nvvm_path = nvvm_dll_path

            def _patched_get_cudalib(lib, static=False):
                if lib == 'nvvm':
                    return _nvvm_path
                return _original_get_cudalib(lib, static=static)
            _cuda_libs.get_cudalib = _patched_get_cudalib

        if cudart_dll_path:
            _original_open_cudalib = _cuda_libs.open_cudalib
            _cudart_path = cudart_dll_path

            def _patched_open_cudalib(lib):
                if lib == 'cudart':
                    return ctypes.CDLL(_cudart_path)
                return ctypes.CDLL(_cuda_libs.get_cudalib(lib))
            _cuda_libs.open_cudalib = _patched_open_cudalib

        if libdevice_path:
            _cuda_libs.get_libdevice = lambda: libdevice_path
            _cuda_libs.open_libdevice = lambda: open(libdevice_path, 'rb').read()
    except Exception:
        pass

    # --- Add CC 12.0 (Blackwell) and CUDA 12.9+ to Numba's support tables ---
    try:
        import numba.cuda.cudadrv.nvvm as _nvvm

        if (12, 0) not in _nvvm.COMPUTE_CAPABILITIES:
            _nvvm.COMPUTE_CAPABILITIES = _nvvm.COMPUTE_CAPABILITIES + ((10, 0), (10, 1), (12, 0),)

        for major in [12, 13]:
            for minor in range(0, 10):
                if (major, minor) not in _nvvm.CTK_SUPPORTED:
                    _nvvm.CTK_SUPPORTED[(major, minor)] = ((5, 0), (12, 0))

        _original_get_arch = _nvvm.get_arch_option
        def _patched_get_arch(major, minor):
            try:
                return _original_get_arch(major, minor)
            except _nvvm.NvvmSupportError:
                return 'compute_90'
        _nvvm.get_arch_option = _patched_get_arch
    except Exception:
        pass

_setup_cuda_paths()

import numba
from numba import cuda

# ============================================
# CUDA DEVICE FUNCTIONS
# ============================================

@cuda.jit(device=True)
def decode_config_cuda(cfg):
    """Decode config_id into individual bit fields.
    Returns tuple of (exit_mask, entry_mask, div_entry_mode, div_exit,
                       div_type, div_ind_mask, cancel_tf, use_ts,
                       reg_inv, hid_inv)
    """
    exit_mask = cfg & 0xF
    entry_mask = (cfg >> 4) & 0x1F
    div_entry_mode = (cfg >> 9) & 0x3
    div_exit = (cfg >> 11) & 0x1
    div_type = (cfg >> 12) & 0x3
    div_ind_mask = (cfg >> 14) & 0xFF
    cancel_tf = (cfg >> 22) & 0x1
    use_ts = (cfg >> 23) & 0x1
    reg_inv = (cfg >> 24) & 0x1
    hid_inv = (cfg >> 25) & 0x1
    return (exit_mask, entry_mask, div_entry_mode, div_exit,
            div_type, div_ind_mask, cancel_tf, use_ts,
            reg_inv, hid_inv)


# ============================================
# CUDA KERNEL - Simulation engine
# ============================================

@cuda.jit
def simulate_kernel(
    configs,            # uint32[N]
    close_arr,          # float64[M]
    high_arr,           # float64[M]
    low_arr,            # float64[M]
    timestamps_i64,     # int64[M]
    zone_bull_arr,      # bool[M]
    zone_bear_arr,      # bool[M]
    filters_forming_arr,   # uint32[M]
    filters_resolved_arr,  # uint32[M]
    div_bits_arr,       # uint8[M, 8]
    sl_pct,             # float64
    sl_emergency_pct,   # float64
    ts_pct,             # float64
    cooldown_bars,      # int32
    commission_pct,     # float64
    n_bars,             # int32 - actual number of bars
    accounting_start,   # int32 - bar index from which to count stats
    results             # float64[N, 7] - output
):
    """Each thread simulates one config over the entire bar range.
    Identical logic to run_simulation_numba in the CPU lab.
    """
    c = cuda.grid(1)
    n_configs = configs.shape[0]
    if c >= n_configs:
        return

    cfg = configs[c]

    # Decode config
    exit_mask = cfg & 0xF
    entry_mask = (cfg >> 4) & 0x1F
    div_entry_mode = (cfg >> 9) & 0x3
    div_exit = (cfg >> 11) & 0x1
    div_type = (cfg >> 12) & 0x3
    div_ind_mask = (cfg >> 14) & 0xFF
    cancel_tf = (cfg >> 22) & 0x1
    use_ts = (cfg >> 23) & 0x1
    reg_inv = (cfg >> 24) & 0x1
    hid_inv = (cfg >> 25) & 0x1

    # State
    position = 0
    entry_price = 0.0
    entry_bar = 0
    entry_filters_forming = 0

    pnl = 0.0
    trades = 0
    wins = 0
    cancels = 0
    peak_pnl = 0.0
    max_dd = 0.0
    gross_profit = 0.0
    gross_loss = 0.0

    div_ctx_bull = False
    div_ctx_bear = False
    last_zone_bull = False

    # Pine-faithful: entries use div from PREVIOUS bar
    prev_div_bull_now = False
    prev_div_bear_now = False

    # Saved div from previous bar
    div_bull_now_saved = False
    div_bear_now_saved = False

    cooldown_until = 0
    sl_level = 0.0

    # Count entry/exit TFs
    entry_tf_count = 0
    for bit in range(5):
        if (entry_mask >> bit) & 1:
            entry_tf_count += 1

    exit_tf_count = 0
    for bit in range(4):
        if (exit_mask >> bit) & 1:
            exit_tf_count += 1

    acct_start = accounting_start

    for t in range(1, n_bars):
        z_bull = zone_bull_arr[t]
        z_bear = zone_bear_arr[t]
        f_forming = filters_forming_arr[t]
        f_resolved = filters_resolved_arr[t]

        close_p = close_arr[t]
        high_p = high_arr[t]
        low_p = low_arr[t]

        # Pine-faithful: save previous bar's div state for entry evaluation
        prev_div_bull_now = div_bull_now_saved
        prev_div_bear_now = div_bear_now_saved

        # Phase 1: Zone change resets
        zone_changed_to_bear = z_bear and last_zone_bull
        zone_changed_to_bull = z_bull and (not last_zone_bull)

        if zone_changed_to_bear:
            div_ctx_bull = False
        if zone_changed_to_bull:
            div_ctx_bear = False

        # Phase 2: div_ctx update from PREVIOUS bar's div_raw
        if prev_div_bull_now:
            div_ctx_bull = True
            div_ctx_bear = False
        if prev_div_bear_now:
            div_ctx_bear = True
            div_ctx_bull = False

        # Snapshot div_ctx for entry evaluation
        entry_div_ctx_bull = div_ctx_bull
        entry_div_ctx_bear = div_ctx_bear

        last_zone_bull = z_bull

        # Phase 3: Calculate divergence for CURRENT bar
        div_bull_now = False
        div_bear_now = False

        if div_type > 0 and div_ind_mask > 0:
            net_div_score = 0
            for ind in range(8):
                if not ((div_ind_mask >> ind) & 1):
                    continue
                bits = div_bits_arr[t, ind]
                ind_bull = False
                ind_bear = False

                if div_type == 1:  # REGULAR only
                    if reg_inv == 0:
                        ind_bull = (bits & 1) > 0
                        ind_bear = (bits & 4) > 0
                    else:
                        ind_bull = (bits & 4) > 0
                        ind_bear = (bits & 1) > 0
                elif div_type == 2:  # HIDDEN only
                    if hid_inv == 0:
                        ind_bull = (bits & 8) > 0
                        ind_bear = (bits & 2) > 0
                    else:
                        ind_bull = (bits & 2) > 0
                        ind_bear = (bits & 8) > 0
                elif div_type == 3:  # BOTH
                    if reg_inv == 0:
                        reg_bull = (bits & 1) > 0
                        reg_bear = (bits & 4) > 0
                    else:
                        reg_bull = (bits & 4) > 0
                        reg_bear = (bits & 1) > 0
                    if hid_inv == 0:
                        hid_bull = (bits & 8) > 0
                        hid_bear = (bits & 2) > 0
                    else:
                        hid_bull = (bits & 2) > 0
                        hid_bear = (bits & 8) > 0
                    ind_bull = reg_bull or hid_bull
                    ind_bear = reg_bear or hid_bear

                if ind_bull:
                    net_div_score += 1
                if ind_bear:
                    net_div_score -= 1

            div_bull_now = net_div_score >= 1
            div_bear_now = net_div_score <= -1

        # Save current bar's div_now for next bar's entry evaluation
        div_bull_now_saved = div_bull_now
        div_bear_now_saved = div_bear_now

        # Phase 4: Update div_ctx with CURRENT bar's div
        if div_bull_now:
            div_ctx_bull = True
        if div_bear_now:
            div_ctx_bear = True

        # ============================================
        # EXIT LOGIC (when in position)
        # ============================================
        if position != 0 and t >= acct_start:
            exit_signal = False
            cancel_signal = False
            div_exit_signal = False
            sl_exit_signal = False
            sl_emergency_signal = False
            normal_exit_signal = False
            exit_price = close_p

            # Trailing stop update
            if use_ts == 1 and t > entry_bar:
                prev_low = low_arr[t - 1]
                prev_high = high_arr[t - 1]
                if position == 1:
                    potential_stop = prev_low * (1.0 - ts_pct / 100.0)
                    if potential_stop > sl_level:
                        sl_level = potential_stop
                elif position == -1:
                    potential_stop = prev_high * (1.0 + ts_pct / 100.0)
                    if sl_level == 0.0 or potential_stop < sl_level:
                        sl_level = potential_stop

            # Emergency SL check
            if position == 1:
                emerg_level = entry_price * (1.0 - sl_emergency_pct / 100.0)
                if low_p <= emerg_level:
                    exit_signal = True
                    sl_exit_signal = True
                    sl_emergency_signal = True
                    exit_price = emerg_level
            elif position == -1:
                emerg_level = entry_price * (1.0 + sl_emergency_pct / 100.0)
                if high_p >= emerg_level:
                    exit_signal = True
                    sl_exit_signal = True
                    sl_emergency_signal = True
                    exit_price = emerg_level

            # Normal SL/TS check
            if (not exit_signal) and sl_level > 0.0:
                if position == 1 and close_p < sl_level:
                    exit_signal = True
                    sl_exit_signal = True
                elif position == -1 and close_p > sl_level:
                    exit_signal = True
                    sl_exit_signal = True

            # Div exit
            if (not exit_signal) and div_exit == 1 and div_type > 0:
                if position == 1 and div_bear_now:
                    exit_signal = True
                    div_exit_signal = True
                elif position == -1 and div_bull_now:
                    exit_signal = True
                    div_exit_signal = True

            # TF exit
            if (not exit_signal) and exit_mask > 0:
                exit_count_bull = 0
                exit_count_active = 0
                for bit in range(4):
                    if (exit_mask >> bit) & 1:
                        exit_count_active += 1
                        if (f_forming >> bit) & 1:
                            exit_count_bull += 1

                if position == 1 and exit_count_active > 0 and exit_count_bull == 0:
                    exit_signal = True
                    normal_exit_signal = True
                elif position == -1 and exit_count_active > 0 and exit_count_bull == exit_count_active:
                    exit_signal = True
                    normal_exit_signal = True

            # Zone exit
            if not exit_signal:
                if position == 1 and z_bear:
                    exit_signal = True
                    normal_exit_signal = True
                elif position == -1 and z_bull:
                    exit_signal = True
                    normal_exit_signal = True

            # Cancel TF
            if (not exit_signal) and cancel_tf == 1:
                cancel_signal = False
                ts_entry_i = timestamps_i64[entry_bar]
                ts_now_i = timestamps_i64[t]
                entry_day = ts_entry_i // 86400000
                current_day = ts_now_i // 86400000
                same_daily = (entry_day == current_day)

                eff = entry_filters_forming
                f_now = filters_forming_arr[t]
                efr = filters_resolved_arr[t]

                if (entry_mask >> 1) & 1:
                    entry_4h = (ts_entry_i // 3600000) // 4
                    now_4h = (ts_now_i // 3600000) // 4
                    if entry_4h == now_4h:
                        if ((eff >> 1) & 1) != ((f_now >> 1) & 1):
                            cancel_signal = True
                    else:
                        if ((eff >> 1) & 1) != ((efr >> 1) & 1):
                            cancel_signal = True

                if (not cancel_signal) and ((entry_mask >> 2) & 1):
                    if same_daily:
                        if ((eff >> 2) & 1) != ((f_now >> 2) & 1):
                            cancel_signal = True
                    else:
                        if ((eff >> 2) & 1) != ((efr >> 2) & 1):
                            cancel_signal = True

            # Process exit
            if exit_signal or cancel_signal:
                if position == 1:
                    trade_pnl = (exit_price - entry_price) / entry_price * 100.0
                else:
                    trade_pnl = (entry_price - exit_price) / entry_price * 100.0

                trade_pnl -= commission_pct

                pnl += trade_pnl
                trades += 1
                if trade_pnl > 0.0:
                    wins += 1
                    gross_profit += trade_pnl
                else:
                    gross_loss += abs(trade_pnl)
                if cancel_signal:
                    cancels += 1

                if pnl > peak_pnl:
                    peak_pnl = pnl
                dd = peak_pnl - pnl
                if dd > max_dd:
                    max_dd = dd

                if position == 1:
                    div_ctx_bull = False
                else:
                    div_ctx_bear = False

                if sl_emergency_signal:
                    cooldown_until = t
                elif sl_exit_signal:
                    cooldown_until = t + cooldown_bars - 1
                elif div_exit_signal:
                    cooldown_until = t + cooldown_bars - 1
                elif cancel_signal:
                    cooldown_until = t

                position = 0
                entry_price = 0.0
                sl_level = 0.0
                entry_filters_forming = 0

        # Post-exit zone div reset (same as CPU)
        if z_bear:
            div_ctx_bull = False
        if z_bull:
            div_ctx_bear = False

        # ============================================
        # ENTRY LOGIC (when flat)
        # ============================================
        if position == 0 and t >= acct_start:
            if t <= cooldown_until:
                continue

            long_cond = False
            short_cond = False

            tf_entry_ok_bull = True
            tf_entry_ok_bear = True

            if (entry_mask >> 0) & 1:
                if not ((f_forming >> 0) & 1):
                    tf_entry_ok_bull = False
                if (f_forming >> 0) & 1:
                    tf_entry_ok_bear = False
            if (entry_mask >> 1) & 1:
                if not ((f_forming >> 1) & 1):
                    tf_entry_ok_bull = False
                if (f_forming >> 1) & 1:
                    tf_entry_ok_bear = False
            if (entry_mask >> 2) & 1:
                if not ((f_forming >> 2) & 1):
                    tf_entry_ok_bull = False
                if (f_forming >> 2) & 1:
                    tf_entry_ok_bear = False
            if (entry_mask >> 3) & 1:
                if not ((f_forming >> 3) & 1):
                    tf_entry_ok_bull = False
                if not ((f_forming >> 11) & 1):
                    tf_entry_ok_bear = False
            if (entry_mask >> 4) & 1:
                if not ((f_forming >> 4) & 1):
                    tf_entry_ok_bull = False
                if not ((f_forming >> 12) & 1):
                    tf_entry_ok_bear = False

            # Pine-faithful: entries use div state from PREVIOUS bar
            effective_ctx_bull = entry_div_ctx_bull or prev_div_bull_now
            effective_ctx_bear = entry_div_ctx_bear or prev_div_bear_now

            if div_entry_mode == 0:
                if z_bull:
                    if entry_tf_count > 0:
                        long_cond = tf_entry_ok_bull
                    else:
                        long_cond = True
                if z_bear:
                    if entry_tf_count > 0:
                        short_cond = tf_entry_ok_bear
                    else:
                        short_cond = True

            elif div_entry_mode == 1:
                if z_bull:
                    if entry_tf_count > 0:
                        long_cond = tf_entry_ok_bull and effective_ctx_bull
                    elif exit_tf_count > 0:
                        long_cond = effective_ctx_bull
                    else:
                        long_cond = prev_div_bull_now

                if z_bear:
                    if entry_tf_count > 0:
                        short_cond = tf_entry_ok_bear and effective_ctx_bear
                    elif exit_tf_count > 0:
                        short_cond = effective_ctx_bear
                    else:
                        short_cond = prev_div_bear_now

            elif div_entry_mode == 2:
                if z_bull:
                    if entry_tf_count > 0:
                        long_cond = tf_entry_ok_bull or prev_div_bull_now
                    else:
                        long_cond = True

                if z_bear:
                    if entry_tf_count > 0:
                        short_cond = tf_entry_ok_bear or prev_div_bear_now
                    else:
                        short_cond = True

            # Exit filter for entries
            if long_cond and exit_mask > 0:
                exit_count_bull = 0
                for bit in range(4):
                    if (exit_mask >> bit) & 1 and (f_forming >> bit) & 1:
                        exit_count_bull += 1
                if exit_count_bull == 0:
                    long_cond = False

            if short_cond and exit_mask > 0:
                exit_count_bull = 0
                exit_count_active = 0
                for bit in range(4):
                    if (exit_mask >> bit) & 1:
                        exit_count_active += 1
                        if (f_forming >> bit) & 1:
                            exit_count_bull += 1
                if exit_count_bull == exit_count_active:
                    short_cond = False

            if long_cond:
                position = 1
                entry_price = close_p
                entry_bar = t
                entry_filters_forming = f_forming
                sl_level = low_p * (1.0 - sl_pct / 100.0)
            elif short_cond:
                position = -1
                entry_price = close_p
                entry_bar = t
                entry_filters_forming = f_forming
                sl_level = high_p * (1.0 + sl_pct / 100.0)

    # Write results
    results[c, 0] = pnl
    results[c, 1] = float(trades)
    results[c, 2] = float(wins)
    results[c, 3] = float(cancels)
    results[c, 4] = max_dd
    results[c, 5] = gross_profit
    results[c, 6] = gross_loss


# ============================================
# CLUSTERED KERNEL - Simulation with per-cluster accounting
# ============================================

MAX_CLUSTERS_CUDA = 32  # compile-time constant for cuda.local.array

@cuda.jit
def simulate_kernel_clustered(
    configs,            # uint32[N]
    close_arr,          # float64[M]
    high_arr,           # float64[M]
    low_arr,            # float64[M]
    timestamps_i64,     # int64[M]
    zone_bull_arr,      # bool[M]
    zone_bear_arr,      # bool[M]
    filters_forming_arr,   # uint32[M]
    filters_resolved_arr,  # uint32[M]
    div_bits_arr,       # uint8[M, 8]
    sl_pct,             # float64
    sl_emergency_pct,   # float64
    ts_pct,             # float64
    cooldown_bars,      # int32
    commission_pct,     # float64
    n_bars,             # int32 - actual number of bars
    accounting_start,   # int32 - bar index from which to count stats
    cluster_labels,     # int64[M] - cluster ID per bar (-1 = unlabeled)
    n_clusters,         # int32 - number of clusters
    results,            # float64[N, 7] - global output
    cl_pnl_out,         # float64[N, n_clusters]
    cl_trades_out,      # int32[N, n_clusters]   (M3 v18 dtype reduction)
    cl_wins_out,        # int32[N, n_clusters]   (M3 v18 dtype reduction)
    cl_maxdd_out,       # float64[N, n_clusters]
    cl_gp_out,          # float64[N, n_clusters]
    cl_gl_out,          # float64[N, n_clusters]
):
    """Each thread simulates one config with per-cluster accounting.
    Identical trading logic to simulate_kernel, plus cluster bookkeeping
    matching run_simulation_numba in the CPU lab.
    """
    c = cuda.grid(1)
    n_configs = configs.shape[0]
    if c >= n_configs:
        return

    cfg = configs[c]

    # Decode config
    exit_mask = cfg & 0xF
    entry_mask = (cfg >> 4) & 0x1F
    div_entry_mode = (cfg >> 9) & 0x3
    div_exit = (cfg >> 11) & 0x1
    div_type = (cfg >> 12) & 0x3
    div_ind_mask = (cfg >> 14) & 0xFF
    cancel_tf = (cfg >> 22) & 0x1
    use_ts = (cfg >> 23) & 0x1
    reg_inv = (cfg >> 24) & 0x1
    hid_inv = (cfg >> 25) & 0x1

    # State
    position = 0
    entry_price = 0.0
    entry_bar = 0
    entry_filters_forming = 0

    pnl = 0.0
    trades = 0
    wins = 0
    cancels = 0
    peak_pnl = 0.0
    max_dd = 0.0
    gross_profit = 0.0
    gross_loss = 0.0

    div_ctx_bull = False
    div_ctx_bear = False
    last_zone_bull = False

    prev_div_bull_now = False
    prev_div_bear_now = False
    div_bull_now_saved = False
    div_bear_now_saved = False

    cooldown_until = 0
    sl_level = 0.0

    entry_tf_count = 0
    for bit in range(5):
        if (entry_mask >> bit) & 1:
            entry_tf_count += 1

    exit_tf_count = 0
    for bit in range(4):
        if (exit_mask >> bit) & 1:
            exit_tf_count += 1

    acct_start = accounting_start

    # Per-cluster local accumulators
    lcl_pnl = cuda.local.array(MAX_CLUSTERS_CUDA, numba.float64)
    lcl_trades = cuda.local.array(MAX_CLUSTERS_CUDA, numba.float64)
    lcl_wins = cuda.local.array(MAX_CLUSTERS_CUDA, numba.float64)
    lcl_maxdd = cuda.local.array(MAX_CLUSTERS_CUDA, numba.float64)
    lcl_gp = cuda.local.array(MAX_CLUSTERS_CUDA, numba.float64)
    lcl_gl = cuda.local.array(MAX_CLUSTERS_CUDA, numba.float64)
    lcl_peak = cuda.local.array(MAX_CLUSTERS_CUDA, numba.float64)

    for k in range(n_clusters):
        lcl_pnl[k] = 0.0
        lcl_trades[k] = 0.0
        lcl_wins[k] = 0.0
        lcl_maxdd[k] = 0.0
        lcl_gp[k] = 0.0
        lcl_gl[k] = 0.0
        lcl_peak[k] = 0.0

    has_clusters = n_clusters > 1

    for t in range(1, n_bars):
        z_bull = zone_bull_arr[t]
        z_bear = zone_bear_arr[t]
        f_forming = filters_forming_arr[t]
        f_resolved = filters_resolved_arr[t]

        close_p = close_arr[t]
        high_p = high_arr[t]
        low_p = low_arr[t]

        prev_div_bull_now = div_bull_now_saved
        prev_div_bear_now = div_bear_now_saved

        # Phase 1: Zone change resets
        zone_changed_to_bear = z_bear and last_zone_bull
        zone_changed_to_bull = z_bull and (not last_zone_bull)

        if zone_changed_to_bear:
            div_ctx_bull = False
        if zone_changed_to_bull:
            div_ctx_bear = False

        # Phase 2: div_ctx update from PREVIOUS bar's div_raw
        if prev_div_bull_now:
            div_ctx_bull = True
            div_ctx_bear = False
        if prev_div_bear_now:
            div_ctx_bear = True
            div_ctx_bull = False

        entry_div_ctx_bull = div_ctx_bull
        entry_div_ctx_bear = div_ctx_bear

        last_zone_bull = z_bull

        # Phase 3: Calculate divergence for CURRENT bar
        div_bull_now = False
        div_bear_now = False

        if div_type > 0 and div_ind_mask > 0:
            net_div_score = 0
            for ind in range(8):
                if not ((div_ind_mask >> ind) & 1):
                    continue
                bits = div_bits_arr[t, ind]
                ind_bull = False
                ind_bear = False

                if div_type == 1:
                    if reg_inv == 0:
                        ind_bull = (bits & 1) > 0
                        ind_bear = (bits & 4) > 0
                    else:
                        ind_bull = (bits & 4) > 0
                        ind_bear = (bits & 1) > 0
                elif div_type == 2:
                    if hid_inv == 0:
                        ind_bull = (bits & 8) > 0
                        ind_bear = (bits & 2) > 0
                    else:
                        ind_bull = (bits & 2) > 0
                        ind_bear = (bits & 8) > 0
                elif div_type == 3:
                    if reg_inv == 0:
                        reg_bull = (bits & 1) > 0
                        reg_bear = (bits & 4) > 0
                    else:
                        reg_bull = (bits & 4) > 0
                        reg_bear = (bits & 1) > 0
                    if hid_inv == 0:
                        hid_bull = (bits & 8) > 0
                        hid_bear = (bits & 2) > 0
                    else:
                        hid_bull = (bits & 2) > 0
                        hid_bear = (bits & 8) > 0
                    ind_bull = reg_bull or hid_bull
                    ind_bear = reg_bear or hid_bear

                if ind_bull:
                    net_div_score += 1
                if ind_bear:
                    net_div_score -= 1

            div_bull_now = net_div_score >= 1
            div_bear_now = net_div_score <= -1

        div_bull_now_saved = div_bull_now
        div_bear_now_saved = div_bear_now

        if div_bull_now:
            div_ctx_bull = True
        if div_bear_now:
            div_ctx_bear = True

        # ============================================
        # EXIT LOGIC (when in position)
        # ============================================
        if position != 0 and t >= acct_start:
            exit_signal = False
            cancel_signal = False
            div_exit_signal = False
            sl_exit_signal = False
            sl_emergency_signal = False
            normal_exit_signal = False
            exit_price = close_p

            if use_ts == 1 and t > entry_bar:
                prev_low = low_arr[t - 1]
                prev_high = high_arr[t - 1]
                if position == 1:
                    potential_stop = prev_low * (1.0 - ts_pct / 100.0)
                    if potential_stop > sl_level:
                        sl_level = potential_stop
                elif position == -1:
                    potential_stop = prev_high * (1.0 + ts_pct / 100.0)
                    if sl_level == 0.0 or potential_stop < sl_level:
                        sl_level = potential_stop

            if position == 1:
                emerg_level = entry_price * (1.0 - sl_emergency_pct / 100.0)
                if low_p <= emerg_level:
                    exit_signal = True
                    sl_exit_signal = True
                    sl_emergency_signal = True
                    exit_price = emerg_level
            elif position == -1:
                emerg_level = entry_price * (1.0 + sl_emergency_pct / 100.0)
                if high_p >= emerg_level:
                    exit_signal = True
                    sl_exit_signal = True
                    sl_emergency_signal = True
                    exit_price = emerg_level

            if (not exit_signal) and sl_level > 0.0:
                if position == 1 and close_p < sl_level:
                    exit_signal = True
                    sl_exit_signal = True
                elif position == -1 and close_p > sl_level:
                    exit_signal = True
                    sl_exit_signal = True

            if (not exit_signal) and div_exit == 1 and div_type > 0:
                if position == 1 and div_bear_now:
                    exit_signal = True
                    div_exit_signal = True
                elif position == -1 and div_bull_now:
                    exit_signal = True
                    div_exit_signal = True

            if (not exit_signal) and exit_mask > 0:
                exit_count_bull = 0
                exit_count_active = 0
                for bit in range(4):
                    if (exit_mask >> bit) & 1:
                        exit_count_active += 1
                        if (f_forming >> bit) & 1:
                            exit_count_bull += 1

                if position == 1 and exit_count_active > 0 and exit_count_bull == 0:
                    exit_signal = True
                    normal_exit_signal = True
                elif position == -1 and exit_count_active > 0 and exit_count_bull == exit_count_active:
                    exit_signal = True
                    normal_exit_signal = True

            if not exit_signal:
                if position == 1 and z_bear:
                    exit_signal = True
                    normal_exit_signal = True
                elif position == -1 and z_bull:
                    exit_signal = True
                    normal_exit_signal = True

            if (not exit_signal) and cancel_tf == 1:
                cancel_signal = False
                ts_entry_i = timestamps_i64[entry_bar]
                ts_now_i = timestamps_i64[t]
                entry_day = ts_entry_i // 86400000
                current_day = ts_now_i // 86400000
                same_daily = (entry_day == current_day)

                eff = entry_filters_forming
                f_now = filters_forming_arr[t]
                efr = filters_resolved_arr[t]

                if (entry_mask >> 1) & 1:
                    entry_4h = (ts_entry_i // 3600000) // 4
                    now_4h = (ts_now_i // 3600000) // 4
                    if entry_4h == now_4h:
                        if ((eff >> 1) & 1) != ((f_now >> 1) & 1):
                            cancel_signal = True
                    else:
                        if ((eff >> 1) & 1) != ((efr >> 1) & 1):
                            cancel_signal = True

                if (not cancel_signal) and ((entry_mask >> 2) & 1):
                    if same_daily:
                        if ((eff >> 2) & 1) != ((f_now >> 2) & 1):
                            cancel_signal = True
                    else:
                        if ((eff >> 2) & 1) != ((efr >> 2) & 1):
                            cancel_signal = True

            # Process exit
            if exit_signal or cancel_signal:
                if position == 1:
                    trade_pnl = (exit_price - entry_price) / entry_price * 100.0
                else:
                    trade_pnl = (entry_price - exit_price) / entry_price * 100.0

                trade_pnl -= commission_pct

                pnl += trade_pnl
                trades += 1
                if trade_pnl > 0.0:
                    wins += 1
                    gross_profit += trade_pnl
                else:
                    gross_loss += abs(trade_pnl)
                if cancel_signal:
                    cancels += 1

                if pnl > peak_pnl:
                    peak_pnl = pnl
                dd = peak_pnl - pnl
                if dd > max_dd:
                    max_dd = dd

                # --- Cluster accounting ---
                if has_clusters:
                    cl_idx = cluster_labels[entry_bar]
                    if 0 <= cl_idx < n_clusters:
                        lcl_pnl[cl_idx] += trade_pnl
                        lcl_trades[cl_idx] += 1.0
                        if trade_pnl > 0.0:
                            lcl_wins[cl_idx] += 1.0
                            lcl_gp[cl_idx] += trade_pnl
                        else:
                            lcl_gl[cl_idx] += abs(trade_pnl)
                        if lcl_pnl[cl_idx] > lcl_peak[cl_idx]:
                            lcl_peak[cl_idx] = lcl_pnl[cl_idx]
                        cl_dd = lcl_peak[cl_idx] - lcl_pnl[cl_idx]
                        if cl_dd > lcl_maxdd[cl_idx]:
                            lcl_maxdd[cl_idx] = cl_dd
                # --- end cluster accounting ---

                if position == 1:
                    div_ctx_bull = False
                else:
                    div_ctx_bear = False

                if sl_emergency_signal:
                    cooldown_until = t
                elif sl_exit_signal:
                    cooldown_until = t + cooldown_bars - 1
                elif div_exit_signal:
                    cooldown_until = t + cooldown_bars - 1
                elif cancel_signal:
                    cooldown_until = t

                position = 0
                entry_price = 0.0
                sl_level = 0.0
                entry_filters_forming = 0

        # Post-exit zone div reset
        if z_bear:
            div_ctx_bull = False
        if z_bull:
            div_ctx_bear = False

        # ============================================
        # ENTRY LOGIC (when flat)
        # ============================================
        if position == 0 and t >= acct_start:
            if t <= cooldown_until:
                continue

            long_cond = False
            short_cond = False

            tf_entry_ok_bull = True
            tf_entry_ok_bear = True

            if (entry_mask >> 0) & 1:
                if not ((f_forming >> 0) & 1):
                    tf_entry_ok_bull = False
                if (f_forming >> 0) & 1:
                    tf_entry_ok_bear = False
            if (entry_mask >> 1) & 1:
                if not ((f_forming >> 1) & 1):
                    tf_entry_ok_bull = False
                if (f_forming >> 1) & 1:
                    tf_entry_ok_bear = False
            if (entry_mask >> 2) & 1:
                if not ((f_forming >> 2) & 1):
                    tf_entry_ok_bull = False
                if (f_forming >> 2) & 1:
                    tf_entry_ok_bear = False
            if (entry_mask >> 3) & 1:
                if not ((f_forming >> 3) & 1):
                    tf_entry_ok_bull = False
                if not ((f_forming >> 11) & 1):
                    tf_entry_ok_bear = False
            if (entry_mask >> 4) & 1:
                if not ((f_forming >> 4) & 1):
                    tf_entry_ok_bull = False
                if not ((f_forming >> 12) & 1):
                    tf_entry_ok_bear = False

            effective_ctx_bull = entry_div_ctx_bull or prev_div_bull_now
            effective_ctx_bear = entry_div_ctx_bear or prev_div_bear_now

            if div_entry_mode == 0:
                if z_bull:
                    if entry_tf_count > 0:
                        long_cond = tf_entry_ok_bull
                    else:
                        long_cond = True
                if z_bear:
                    if entry_tf_count > 0:
                        short_cond = tf_entry_ok_bear
                    else:
                        short_cond = True

            elif div_entry_mode == 1:
                if z_bull:
                    if entry_tf_count > 0:
                        long_cond = tf_entry_ok_bull and effective_ctx_bull
                    elif exit_tf_count > 0:
                        long_cond = effective_ctx_bull
                    else:
                        long_cond = prev_div_bull_now

                if z_bear:
                    if entry_tf_count > 0:
                        short_cond = tf_entry_ok_bear and effective_ctx_bear
                    elif exit_tf_count > 0:
                        short_cond = effective_ctx_bear
                    else:
                        short_cond = prev_div_bear_now

            elif div_entry_mode == 2:
                if z_bull:
                    if entry_tf_count > 0:
                        long_cond = tf_entry_ok_bull or prev_div_bull_now
                    else:
                        long_cond = True

                if z_bear:
                    if entry_tf_count > 0:
                        short_cond = tf_entry_ok_bear or prev_div_bear_now
                    else:
                        short_cond = True

            if long_cond and exit_mask > 0:
                exit_count_bull = 0
                for bit in range(4):
                    if (exit_mask >> bit) & 1 and (f_forming >> bit) & 1:
                        exit_count_bull += 1
                if exit_count_bull == 0:
                    long_cond = False

            if short_cond and exit_mask > 0:
                exit_count_bull = 0
                exit_count_active = 0
                for bit in range(4):
                    if (exit_mask >> bit) & 1:
                        exit_count_active += 1
                        if (f_forming >> bit) & 1:
                            exit_count_bull += 1
                if exit_count_bull == exit_count_active:
                    short_cond = False

            if long_cond:
                position = 1
                entry_price = close_p
                entry_bar = t
                entry_filters_forming = f_forming
                sl_level = low_p * (1.0 - sl_pct / 100.0)
            elif short_cond:
                position = -1
                entry_price = close_p
                entry_bar = t
                entry_filters_forming = f_forming
                sl_level = high_p * (1.0 + sl_pct / 100.0)

    # Write global results
    results[c, 0] = pnl
    results[c, 1] = float(trades)
    results[c, 2] = float(wins)
    results[c, 3] = float(cancels)
    results[c, 4] = max_dd
    results[c, 5] = gross_profit
    results[c, 6] = gross_loss

    # Write per-cluster results
    # M3 v18 dtype reduction: cl_trades_out + cl_wins_out alocados int32 (saves
    # 50% storage cl_trades + cl_wins). lcl_* stays float64 (kernel arithmetic),
    # cast explícito a int al store site para coerción determinística.
    if has_clusters:
        for k in range(n_clusters):
            cl_pnl_out[c, k] = lcl_pnl[k]
            cl_trades_out[c, k] = int(lcl_trades[k])
            cl_wins_out[c, k] = int(lcl_wins[k])
            cl_maxdd_out[c, k] = lcl_maxdd[k]
            cl_gp_out[c, k] = lcl_gp[k]
            cl_gl_out[c, k] = lcl_gl[k]


# ============================================
# DATA HANDLE - Holds device arrays
# ============================================

class DataHandle:
    """Holds precalculated data arrays in GPU device memory."""

    def __init__(self, close, high, low, timestamps_i64,
                 zone_bull, zone_bear,
                 filters_forming, filters_resolved,
                 div_bits):
        self.close = close
        self.high = high
        self.low = low
        self.timestamps_i64 = timestamps_i64
        self.zone_bull = zone_bull
        self.zone_bear = zone_bear
        self.filters_forming = filters_forming
        self.filters_resolved = filters_resolved
        self.div_bits = div_bits
        self.n_bars = close.shape[0]

    def __repr__(self):
        return f"DataHandle(n_bars={self.n_bars})"


# ============================================
# CUDA SIMULATOR
# ============================================

class CUDASimulator:
    """GPU-accelerated trading simulation engine.

    Usage:
        sim = CUDASimulator()
        handle = sim.upload_data(data_dict)
        results = sim.run_on_slice(configs, handle, start, end, sl, sl_e, ts, cd, comm)
    """

    def __init__(self, threads_per_block=256):
        """Detect GPU and initialize."""
        self.gpu_available = False
        self.threads_per_block = threads_per_block

        try:
            cuda.detect()
            device = cuda.get_current_device()
            self.gpu_name = device.name
            self.gpu_available = True
            # Compute capability
            cc = device.compute_capability
            print(f"[CUDA] GPU detectada: {self.gpu_name} (CC {cc[0]}.{cc[1]})")
            print(f"[CUDA] Threads per block: {threads_per_block}")
        except Exception as e:
            print(f"[CUDA] No se detecto GPU: {e}")
            print(f"[CUDA] Se usara fallback CPU")

    def upload_data(self, data_dict):
        """Copy precalculated arrays to device memory.

        Args:
            data_dict: dict returned by precalculate_all_data with keys:
                close, high, low, timestamps, zone_bull, zone_bear,
                filters_forming, filters_resolved, div_bits

        Returns:
            DataHandle for reuse in multiple run_on_slice calls.
        """
        if not self.gpu_available:
            return data_dict  # Fallback: just return the dict

        # Convert timestamps to int64 milliseconds
        ts_raw = data_dict['timestamps']
        ts_i64 = ts_raw.astype('datetime64[ms]').astype(np.int64)

        # Ensure correct dtypes
        close = np.ascontiguousarray(data_dict['close'], dtype=np.float64)
        high = np.ascontiguousarray(data_dict['high'], dtype=np.float64)
        low = np.ascontiguousarray(data_dict['low'], dtype=np.float64)
        zone_bull = np.ascontiguousarray(data_dict['zone_bull'], dtype=np.bool_)
        zone_bear = np.ascontiguousarray(data_dict['zone_bear'], dtype=np.bool_)
        filters_forming = np.ascontiguousarray(data_dict['filters_forming'], dtype=np.uint32)
        filters_resolved = np.ascontiguousarray(data_dict['filters_resolved'], dtype=np.uint32)
        div_bits = np.ascontiguousarray(data_dict['div_bits'], dtype=np.uint8)

        # Copy to device
        handle = DataHandle(
            close=cuda.to_device(close),
            high=cuda.to_device(high),
            low=cuda.to_device(low),
            timestamps_i64=cuda.to_device(ts_i64),
            zone_bull=cuda.to_device(zone_bull),
            zone_bear=cuda.to_device(zone_bear),
            filters_forming=cuda.to_device(filters_forming),
            filters_resolved=cuda.to_device(filters_resolved),
            div_bits=cuda.to_device(div_bits),
        )

        print(f"[CUDA] Datos subidos a GPU: {handle.n_bars} bars")
        return handle

    def run_on_slice(self, configs, data_handle, start_bar, end_bar,
                     sl_pct, sl_emergency_pct, ts_pct, cooldown_bars, commission_pct,
                     warmup=100):
        """Execute simulation on GPU.

        Same signature as lab.run_on_slice.
        warmup: bars before start_bar used to build state without trading.
        Returns array (N, 7) in host memory with
        [pnl, trades, wins, cancels, maxdd, gross_profit, gross_loss].
        """
        if not self.gpu_available:
            return self.run_on_slice_cpu_fallback(
                configs, data_handle, start_bar, end_bar,
                sl_pct, sl_emergency_pct, ts_pct, cooldown_bars, commission_pct,
                warmup=warmup
            )

        n_configs = len(configs)
        actual_start = max(0, start_bar - warmup)
        accounting_start = start_bar - actual_start
        s, e = actual_start, end_bar
        n_bars_slice = e - s

        # Slice device arrays - we need to copy sliced data
        # Since CUDA device arrays don't support arbitrary slicing well,
        # we slice on host then copy, OR we pass start/end to kernel.
        # Better approach: slice on host, copy slice to device.
        # But if data_handle already has full data, we can use offsets.
        # For simplicity and correctness, let's slice and upload the slice.

        if isinstance(data_handle, dict):
            # Fallback path
            return self.run_on_slice_cpu_fallback(
                configs, data_handle, start_bar, end_bar,
                sl_pct, sl_emergency_pct, ts_pct, cooldown_bars, commission_pct
            )

        # Slice from full device arrays - copy slice to new device arrays
        # We need to do this on host side since CUDA device arrays have
        # limited slicing support
        close_slice = data_handle.close.copy_to_host()[s:e]
        high_slice = data_handle.high.copy_to_host()[s:e]
        low_slice = data_handle.low.copy_to_host()[s:e]
        ts_slice = data_handle.timestamps_i64.copy_to_host()[s:e]
        zone_bull_slice = data_handle.zone_bull.copy_to_host()[s:e]
        zone_bear_slice = data_handle.zone_bear.copy_to_host()[s:e]
        filters_forming_slice = data_handle.filters_forming.copy_to_host()[s:e]
        filters_resolved_slice = data_handle.filters_resolved.copy_to_host()[s:e]
        div_bits_slice = data_handle.div_bits.copy_to_host()[s:e]

        # Upload slices to device
        d_close = cuda.to_device(np.ascontiguousarray(close_slice))
        d_high = cuda.to_device(np.ascontiguousarray(high_slice))
        d_low = cuda.to_device(np.ascontiguousarray(low_slice))
        d_ts = cuda.to_device(np.ascontiguousarray(ts_slice))
        d_zone_bull = cuda.to_device(np.ascontiguousarray(zone_bull_slice))
        d_zone_bear = cuda.to_device(np.ascontiguousarray(zone_bear_slice))
        d_filters_forming = cuda.to_device(np.ascontiguousarray(filters_forming_slice))
        d_filters_resolved = cuda.to_device(np.ascontiguousarray(filters_resolved_slice))
        d_div_bits = cuda.to_device(np.ascontiguousarray(div_bits_slice))

        # Configs to device
        d_configs = cuda.to_device(np.ascontiguousarray(configs, dtype=np.uint32))

        # Output array on device
        d_results = cuda.device_array((n_configs, 7), dtype=np.float64)

        # Launch kernel
        threads_per_block = self.threads_per_block
        blocks_per_grid = (n_configs + threads_per_block - 1) // threads_per_block

        simulate_kernel[blocks_per_grid, threads_per_block](
            d_configs,
            d_close, d_high, d_low, d_ts,
            d_zone_bull, d_zone_bear,
            d_filters_forming, d_filters_resolved,
            d_div_bits,
            sl_pct, sl_emergency_pct, ts_pct, cooldown_bars, commission_pct,
            n_bars_slice,
            accounting_start,
            d_results
        )

        # Synchronize and copy back
        cuda.synchronize()
        results = d_results.copy_to_host()

        return results

    def run_on_slice_cpu_fallback(self, configs, data_handle, start_bar, end_bar,
                                  sl_pct, sl_emergency_pct, ts_pct, cooldown_bars, commission_pct,
                                  warmup=100):
        """Fallback to CPU if GPU not available."""
        import lab_historico_numba_v8_3 as lab
        if isinstance(data_handle, DataHandle):
            # Reconstruct data dict from handle (copy from device)
            data = {
                'close': data_handle.close.copy_to_host(),
                'high': data_handle.high.copy_to_host(),
                'low': data_handle.low.copy_to_host(),
                'timestamps': data_handle.timestamps_i64.copy_to_host(),  # already int64
                'zone_bull': data_handle.zone_bull.copy_to_host(),
                'zone_bear': data_handle.zone_bear.copy_to_host(),
                'filters_forming': data_handle.filters_forming.copy_to_host(),
                'filters_resolved': data_handle.filters_resolved.copy_to_host(),
                'div_bits': data_handle.div_bits.copy_to_host(),
            }
        else:
            data = data_handle
        return lab.run_on_slice(configs, data, start_bar, end_bar,
                               sl_pct, sl_emergency_pct, ts_pct, cooldown_bars, commission_pct,
                               warmup=warmup)

    def run_on_slice_batched(self, configs, data_handle, start_bar, end_bar,
                             sl_pct, sl_emergency_pct, ts_pct, cooldown_bars, commission_pct,
                             batch_size=1_000_000):
        """Run simulation in batches to manage GPU memory.

        For very large config arrays (>1M), splits into batches to avoid
        running out of VRAM on the output array.
        """
        n_configs = len(configs)
        if n_configs <= batch_size:
            return self.run_on_slice(
                configs, data_handle, start_bar, end_bar,
                sl_pct, sl_emergency_pct, ts_pct, cooldown_bars, commission_pct
            )

        # Process in batches
        all_results = np.zeros((n_configs, 7), dtype=np.float64)
        n_batches = (n_configs + batch_size - 1) // batch_size

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_configs)
            batch_configs = configs[start_idx:end_idx]

            batch_results = self.run_on_slice(
                batch_configs, data_handle, start_bar, end_bar,
                sl_pct, sl_emergency_pct, ts_pct, cooldown_bars, commission_pct
            )
            all_results[start_idx:end_idx] = batch_results

        return all_results


# ============================================
# OPTIMIZED VERSION - Avoid host round-trip for slicing
# ============================================

class CUDASimulatorOptimized(CUDASimulator):
    """Optimized version that caches host arrays to avoid device->host->device
    round trips when slicing."""

    def upload_data(self, data_dict):
        """Upload data and also keep host copies for efficient slicing."""
        if not self.gpu_available:
            return data_dict

        # Mitigation v17 post-DIAG.11 cross-2-crashes TDR 0x116 STATUS_INSUFFICIENT_RESOURCES:
        # libera host arrays previos + flush deferred GPU dealloc queue ANTES de
        # re-allocate para nuevo símbolo/preset. Sin esto, cumulative ~1152
        # device arrays de previos handles quedan en pending dealloc.
        if hasattr(self, '_host_close'):
            del self._host_close, self._host_high, self._host_low, self._host_ts
            del self._host_zone_bull, self._host_zone_bear
            del self._host_filters_forming, self._host_filters_resolved
            del self._host_div_bits
        try:
            cuda.current_context().deallocations.clear()
        except Exception:
            pass

        # Convert timestamps to int64 milliseconds
        ts_raw = data_dict['timestamps']
        ts_i64 = ts_raw.astype('datetime64[ms]').astype(np.int64)

        # Keep host copies for slicing
        self._host_close = np.ascontiguousarray(data_dict['close'], dtype=np.float64)
        self._host_high = np.ascontiguousarray(data_dict['high'], dtype=np.float64)
        self._host_low = np.ascontiguousarray(data_dict['low'], dtype=np.float64)
        self._host_ts = np.ascontiguousarray(ts_i64)
        self._host_zone_bull = np.ascontiguousarray(data_dict['zone_bull'], dtype=np.bool_)
        self._host_zone_bear = np.ascontiguousarray(data_dict['zone_bear'], dtype=np.bool_)
        self._host_filters_forming = np.ascontiguousarray(data_dict['filters_forming'], dtype=np.uint32)
        self._host_filters_resolved = np.ascontiguousarray(data_dict['filters_resolved'], dtype=np.uint32)
        self._host_div_bits = np.ascontiguousarray(data_dict['div_bits'], dtype=np.uint8)

        # Upload full arrays to device too (for potential full-range runs)
        handle = DataHandle(
            close=cuda.to_device(self._host_close),
            high=cuda.to_device(self._host_high),
            low=cuda.to_device(self._host_low),
            timestamps_i64=cuda.to_device(self._host_ts),
            zone_bull=cuda.to_device(self._host_zone_bull),
            zone_bear=cuda.to_device(self._host_zone_bear),
            filters_forming=cuda.to_device(self._host_filters_forming),
            filters_resolved=cuda.to_device(self._host_filters_resolved),
            div_bits=cuda.to_device(self._host_div_bits),
        )

        print(f"[CUDA] Datos subidos a GPU: {handle.n_bars} bars (optimized)")
        return handle

    def run_on_slice(self, configs, data_handle, start_bar, end_bar,
                     sl_pct, sl_emergency_pct, ts_pct, cooldown_bars, commission_pct,
                     warmup=100, cluster_labels=None, n_clusters=1, chunk_size=None):
        """Execute simulation on GPU, slicing from cached host arrays.

        When cluster_labels is provided, uses the clustered kernel and returns
        (results, cl_pnl, cl_trades, cl_wins, cl_maxdd, cl_gp, cl_gl)
        matching the CPU run_on_slice signature.
        Otherwise returns just the results array.

        chunk_size (R1 v18): when provided and < n_configs, partitions configs
        en chunks que respetan capacity GDDR7 8GB. None ó >= n_configs → single
        kernel launch (legacy behavior preserved). Solo aplica al clustered path
        (non-clustered path consume <1 GB peak independiente n_configs).

        Nota 2026-05-19: C3 refactor attempt (force chunking + effective_chunk_size
        cap n_configs//2) EMPIRICALLY REFUTED post-Crash 13 cumulative — bug
        nvlddmkm.sys offset _2440 NO solucionable vía Python/Numba code changes
        cross-Sub-Sesiones precedent absoluto. WSL2/Linux native bypass (Opción D
        Sub-Sesión 2026-05-11 F.6) es siguiente path. Ver CONTEXTO §13.4 entry
        Crash 13 + C3 refute 2026-05-19.
        """
        has_clusters = cluster_labels is not None and n_clusters > 1

        if not self.gpu_available or isinstance(data_handle, dict):
            return self._run_on_slice_cpu_with_clusters(
                configs, data_handle, start_bar, end_bar,
                sl_pct, sl_emergency_pct, ts_pct, cooldown_bars, commission_pct,
                warmup=warmup, cluster_labels=cluster_labels, n_clusters=n_clusters
            )

        n_configs = len(configs)
        actual_start = max(0, start_bar - warmup)
        accounting_start = start_bar - actual_start
        s, e = actual_start, end_bar
        n_bars_slice = e - s

        # Slice from cached host arrays (avoids device->host copy) — shared
        # cross chunks (bar-related, ~5 MB total)
        d_close = cuda.to_device(np.ascontiguousarray(self._host_close[s:e]))
        d_high = cuda.to_device(np.ascontiguousarray(self._host_high[s:e]))
        d_low = cuda.to_device(np.ascontiguousarray(self._host_low[s:e]))
        d_ts = cuda.to_device(np.ascontiguousarray(self._host_ts[s:e]))
        d_zone_bull = cuda.to_device(np.ascontiguousarray(self._host_zone_bull[s:e]))
        d_zone_bear = cuda.to_device(np.ascontiguousarray(self._host_zone_bear[s:e]))
        d_filters_forming = cuda.to_device(np.ascontiguousarray(self._host_filters_forming[s:e]))
        d_filters_resolved = cuda.to_device(np.ascontiguousarray(self._host_filters_resolved[s:e]))
        d_div_bits = cuda.to_device(np.ascontiguousarray(self._host_div_bits[s:e]))

        threads_per_block = self.threads_per_block

        if has_clusters:
            if n_clusters > MAX_CLUSTERS_CUDA:
                raise ValueError(f"n_clusters={n_clusters} exceeds MAX_CLUSTERS_CUDA={MAX_CLUSTERS_CUDA}")

            # Slice cluster labels and upload (shared cross chunks)
            cl_labels_slice = np.ascontiguousarray(cluster_labels[s:e], dtype=np.int64)
            d_cl_labels = cuda.to_device(cl_labels_slice)

            # R1 v18: determine chunk strategy. None / >= n_configs → single call.
            use_chunking = (chunk_size is not None
                            and chunk_size > 0
                            and chunk_size < n_configs)

            if not use_chunking:
                # Single-call path (legacy + v17 cleanup pattern preserved)
                d_configs = cuda.to_device(np.ascontiguousarray(configs, dtype=np.uint32))
                d_results = cuda.device_array((n_configs, 7), dtype=np.float64)
                d_cl_pnl = cuda.device_array((n_configs, n_clusters), dtype=np.float64)
                d_cl_trades = cuda.device_array((n_configs, n_clusters), dtype=np.int32)   # M3
                d_cl_wins = cuda.device_array((n_configs, n_clusters), dtype=np.int32)     # M3
                d_cl_maxdd = cuda.device_array((n_configs, n_clusters), dtype=np.float64)
                d_cl_gp = cuda.device_array((n_configs, n_clusters), dtype=np.float64)
                d_cl_gl = cuda.device_array((n_configs, n_clusters), dtype=np.float64)

                blocks_per_grid = (n_configs + threads_per_block - 1) // threads_per_block
                simulate_kernel_clustered[blocks_per_grid, threads_per_block](
                    d_configs,
                    d_close, d_high, d_low, d_ts,
                    d_zone_bull, d_zone_bear,
                    d_filters_forming, d_filters_resolved,
                    d_div_bits,
                    sl_pct, sl_emergency_pct, ts_pct, cooldown_bars, commission_pct,
                    n_bars_slice,
                    accounting_start,
                    d_cl_labels,
                    n_clusters,
                    d_results,
                    d_cl_pnl, d_cl_trades, d_cl_wins, d_cl_maxdd, d_cl_gp, d_cl_gl
                )

                cuda.synchronize()
                results_h = d_results.copy_to_host()
                cl_pnl_h = d_cl_pnl.copy_to_host()
                cl_trades_h = d_cl_trades.copy_to_host()
                cl_wins_h = d_cl_wins.copy_to_host()
                cl_maxdd_h = d_cl_maxdd.copy_to_host()
                cl_gp_h = d_cl_gp.copy_to_host()
                cl_gl_h = d_cl_gl.copy_to_host()
                del d_close, d_high, d_low, d_ts, d_zone_bull, d_zone_bear
                del d_filters_forming, d_filters_resolved, d_div_bits
                del d_configs, d_results
                del d_cl_labels, d_cl_pnl, d_cl_trades, d_cl_wins, d_cl_maxdd, d_cl_gp, d_cl_gl
                try:
                    cuda.current_context().deallocations.clear()
                except Exception:
                    pass
                return (results_h, cl_pnl_h, cl_trades_h, cl_wins_h,
                        cl_maxdd_h, cl_gp_h, cl_gl_h)

            # Chunked path R1 v18 — partitions configs, kernel launch per chunk
            results_h = np.empty((n_configs, 7), dtype=np.float64)
            cl_pnl_h = np.empty((n_configs, n_clusters), dtype=np.float64)
            cl_trades_h = np.empty((n_configs, n_clusters), dtype=np.int32)    # M3
            cl_wins_h = np.empty((n_configs, n_clusters), dtype=np.int32)      # M3
            cl_maxdd_h = np.empty((n_configs, n_clusters), dtype=np.float64)
            cl_gp_h = np.empty((n_configs, n_clusters), dtype=np.float64)
            cl_gl_h = np.empty((n_configs, n_clusters), dtype=np.float64)

            configs_full = np.ascontiguousarray(configs, dtype=np.uint32)
            n_chunks = (n_configs + chunk_size - 1) // chunk_size

            for ci in range(n_chunks):
                cs = ci * chunk_size
                ce = min(cs + chunk_size, n_configs)
                chunk_n = ce - cs

                d_configs_chunk = cuda.to_device(configs_full[cs:ce])
                d_results_chunk = cuda.device_array((chunk_n, 7), dtype=np.float64)
                d_cl_pnl_chunk = cuda.device_array((chunk_n, n_clusters), dtype=np.float64)
                d_cl_trades_chunk = cuda.device_array((chunk_n, n_clusters), dtype=np.int32)   # M3
                d_cl_wins_chunk = cuda.device_array((chunk_n, n_clusters), dtype=np.int32)     # M3
                d_cl_maxdd_chunk = cuda.device_array((chunk_n, n_clusters), dtype=np.float64)
                d_cl_gp_chunk = cuda.device_array((chunk_n, n_clusters), dtype=np.float64)
                d_cl_gl_chunk = cuda.device_array((chunk_n, n_clusters), dtype=np.float64)

                blocks_per_grid_chunk = (chunk_n + threads_per_block - 1) // threads_per_block
                simulate_kernel_clustered[blocks_per_grid_chunk, threads_per_block](
                    d_configs_chunk,
                    d_close, d_high, d_low, d_ts,
                    d_zone_bull, d_zone_bear,
                    d_filters_forming, d_filters_resolved,
                    d_div_bits,
                    sl_pct, sl_emergency_pct, ts_pct, cooldown_bars, commission_pct,
                    n_bars_slice,
                    accounting_start,
                    d_cl_labels,
                    n_clusters,
                    d_results_chunk,
                    d_cl_pnl_chunk, d_cl_trades_chunk, d_cl_wins_chunk,
                    d_cl_maxdd_chunk, d_cl_gp_chunk, d_cl_gl_chunk
                )

                cuda.synchronize()
                results_h[cs:ce] = d_results_chunk.copy_to_host()
                cl_pnl_h[cs:ce] = d_cl_pnl_chunk.copy_to_host()
                cl_trades_h[cs:ce] = d_cl_trades_chunk.copy_to_host()
                cl_wins_h[cs:ce] = d_cl_wins_chunk.copy_to_host()
                cl_maxdd_h[cs:ce] = d_cl_maxdd_chunk.copy_to_host()
                cl_gp_h[cs:ce] = d_cl_gp_chunk.copy_to_host()
                cl_gl_h[cs:ce] = d_cl_gl_chunk.copy_to_host()

                # v17 cleanup pattern extended per-chunk: 8 device arrays freed
                del d_configs_chunk, d_results_chunk
                del d_cl_pnl_chunk, d_cl_trades_chunk, d_cl_wins_chunk
                del d_cl_maxdd_chunk, d_cl_gp_chunk, d_cl_gl_chunk
                try:
                    cuda.current_context().deallocations.clear()
                except Exception:
                    pass

            # Cleanup shared bar-related arrays at end (9 device arrays + cl_labels)
            del d_close, d_high, d_low, d_ts, d_zone_bull, d_zone_bear
            del d_filters_forming, d_filters_resolved, d_div_bits, d_cl_labels
            try:
                cuda.current_context().deallocations.clear()
            except Exception:
                pass

            return (results_h, cl_pnl_h, cl_trades_h, cl_wins_h,
                    cl_maxdd_h, cl_gp_h, cl_gl_h)
        else:
            # Non-clustered path — chunking NOT applied (consume <1 GB peak
            # independent de n_configs, no risk capacity hardware)
            d_configs = cuda.to_device(np.ascontiguousarray(configs, dtype=np.uint32))
            d_results = cuda.device_array((n_configs, 7), dtype=np.float64)
            blocks_per_grid = (n_configs + threads_per_block - 1) // threads_per_block
            simulate_kernel[blocks_per_grid, threads_per_block](
                d_configs,
                d_close, d_high, d_low, d_ts,
                d_zone_bull, d_zone_bear,
                d_filters_forming, d_filters_resolved,
                d_div_bits,
                sl_pct, sl_emergency_pct, ts_pct, cooldown_bars, commission_pct,
                n_bars_slice,
                accounting_start,
                d_results
            )

            cuda.synchronize()
            # Copy to host antes de del (mitigation v17 post-DIAG.11)
            results_h = d_results.copy_to_host()
            # Cleanup explícito 11 device arrays + flush deferred dealloc
            del d_close, d_high, d_low, d_ts, d_zone_bull, d_zone_bear
            del d_filters_forming, d_filters_resolved, d_div_bits
            del d_configs, d_results
            try:
                cuda.current_context().deallocations.clear()
            except Exception:
                pass
            return results_h

    def _run_on_slice_cpu_with_clusters(self, configs, data_handle, start_bar, end_bar,
                                         sl_pct, sl_emergency_pct, ts_pct, cooldown_bars, commission_pct,
                                         warmup=100, cluster_labels=None, n_clusters=1):
        """CPU fallback that supports cluster_labels."""
        import lab_historico_numba_v8_3 as lab
        if isinstance(data_handle, DataHandle):
            data = {
                'close': data_handle.close.copy_to_host(),
                'high': data_handle.high.copy_to_host(),
                'low': data_handle.low.copy_to_host(),
                'timestamps': data_handle.timestamps_i64.copy_to_host(),
                'zone_bull': data_handle.zone_bull.copy_to_host(),
                'zone_bear': data_handle.zone_bear.copy_to_host(),
                'filters_forming': data_handle.filters_forming.copy_to_host(),
                'filters_resolved': data_handle.filters_resolved.copy_to_host(),
                'div_bits': data_handle.div_bits.copy_to_host(),
            }
        else:
            data = data_handle
        return lab.run_on_slice(configs, data, start_bar, end_bar,
                               sl_pct, sl_emergency_pct, ts_pct, cooldown_bars, commission_pct,
                               warmup=warmup, cluster_labels=cluster_labels, n_clusters=n_clusters)
