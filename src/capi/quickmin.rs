//! Quick-min variant of a band (ABI minor 16).

use super::*;

/// Quick-min variant for [`rgsaddle_band_set_quickmin_variant`]: FIRE
/// time-step control with one velocity projection over the whole band. The
/// default of a quick-min band.
pub const RGSADDLE_QUICKMIN_RGMIN: i32 = 0;
/// Quick-min variant for [`rgsaddle_band_set_quickmin_variant`]: eOn's
/// quick-min, with a fixed time step and the velocity kept at its cap.
pub const RGSADDLE_QUICKMIN_EON: i32 = 1;
/// Quick-min variant for [`rgsaddle_band_set_quickmin_variant`]:
/// [`RGSADDLE_QUICKMIN_RGMIN`] with the velocity projection taken per atom.
pub const RGSADDLE_QUICKMIN_PER_ATOM: i32 = 2;

/// Select the quick-min variant of a band made with `RGSADDLE_METHOD_QUICKMIN`.
///
/// `variant` is one of [`RGSADDLE_QUICKMIN_RGMIN`], [`RGSADDLE_QUICKMIN_EON`]
/// and [`RGSADDLE_QUICKMIN_PER_ATOM`]. The velocity and the time step restart.
///
/// Returns `RGSADDLE_OK`, `RGSADDLE_NULL_BAND` for a NULL `band`, and
/// `RGSADDLE_INVALID_PARAMETER` for an unknown `variant` or a band that is not
/// quick-min; the band is unchanged in both of the last two cases.
///
/// # Safety
/// `band` is a live session or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_set_quickmin_variant(
    band: *mut RgsaddleBand,
    variant: i32,
) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    let options = match variant {
        RGSADDLE_QUICKMIN_RGMIN => rgmin::QuickMinOptions::rgmin(),
        RGSADDLE_QUICKMIN_EON => rgmin::QuickMinOptions::eon(),
        RGSADDLE_QUICKMIN_PER_ATOM => rgmin::QuickMinOptions::per_atom(),
        _ => return RGSADDLE_INVALID_PARAMETER,
    };
    let band = unsafe { &mut *band };
    if !band.session.is_quickmin() {
        return RGSADDLE_INVALID_PARAMETER;
    }
    band.session.set_quickmin_options(options);
    RGSADDLE_OK
}

/// Replace the initial time step of the FIRE and quick-min band optimizers.
///
/// `time_step` is the rgmin `Control::istep`, in the optimizer's own step
/// units (1 at creation), not in femtoseconds; [`RGSADDLE_QUICKMIN_EON`] keeps
/// it fixed. The optimizer is rebuilt, so its history and the HiGHS selection
/// go, while cached surface values stay.
///
/// Returns `RGSADDLE_OK`, `RGSADDLE_NULL_BAND` for a NULL `band`, and
/// `RGSADDLE_INVALID_PARAMETER` for a `time_step` that is not positive and
/// finite; the band is unchanged in that case.
///
/// # Safety
/// `band` is a live session or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rgsaddle_band_set_time_step(
    band: *mut RgsaddleBand,
    time_step: f64,
) -> i32 {
    if band.is_null() {
        return RGSADDLE_NULL_BAND;
    }
    match unsafe { (*band).session.set_time_step(time_step) } {
        Ok(()) => RGSADDLE_OK,
        Err(e) => status_of(&e),
    }
}
