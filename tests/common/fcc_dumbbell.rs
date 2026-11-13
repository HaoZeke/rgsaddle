// The FCC `<100>` dumbbell translation rule shared by the copper tests.

/// Axis of a `<100>` dumbbell after a nearest-neighbor translation.
///
/// Eight of the twelve FCC neighbor hops move the center and turn the axis
/// by a right angle. The new axis is the part of `hop` perpendicular to
/// `axis`, and that part lies along one cube edge. The four hops in the
/// plane normal to `axis` are not this mechanism: their perpendicular part
/// is not a `<100>` axis. The returned axis is unsigned, because a dumbbell
/// is unchanged by reversing it.
fn dumbbell_translation_axis(axis: [f64; 3], hop: [f64; 3]) -> Option<[f64; 3]> {
    let axis_norm2 = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
    let hop_norm2 = hop[0] * hop[0] + hop[1] * hop[1] + hop[2] * hop[2];
    if !(axis_norm2.is_finite() && hop_norm2.is_finite())
        || axis_norm2 < 1.0e-18
        || hop_norm2 < 1.0e-18
    {
        return None;
    }
    let dot = axis[0] * hop[0] + axis[1] * hop[1] + axis[2] * hop[2];
    let scale = dot / axis_norm2;
    let perp = [
        hop[0] - axis[0] * scale,
        hop[1] - axis[1] * scale,
        hop[2] - axis[2] * scale,
    ];
    let floor = 1.0e-8 * hop_norm2.sqrt();
    let mut nonzero = 0usize;
    let mut which = 0usize;
    for (component, value) in perp.iter().enumerate() {
        if value.abs() > floor {
            nonzero += 1;
            which = component;
        }
    }
    if nonzero != 1 {
        return None;
    }
    let mut out = [0.0; 3];
    out[which] = 1.0;
    Some(out)
}
