//! One-dimensional velocity Verlet on the cubic signed image force.

use super::*;

/// Integrate the action on the same Simpson grid as the quadrature clock.
/// The force-interpolated period needs no adaptive endpoint time integral.
pub(in crate::lineb) fn action_only<S: PointSurface>(
    surface: &S, masses: &Masses, curve: &Curve, energy: f64, pairs: usize,
) -> Result<f64, SaddleError> {
    let level = Level { surface, masses, curve, energy };
    let mut action = 0.;
    for (j, bounds) in curve.knots.windows(2).enumerate() {
        let (a, b) = (bounds[0], bounds[1]);
        if j == 0 || j+2 == curve.knots.len() {
            action += end_action(&level, EndChord::new(a,b,j==0)?, pairs)?;
        } else {
            let n = 2*pairs;
            let h = (b-a)/n as f64;
            let mut sum = 0.;
            for i in 0..=n {
                let v = level.sample(a + h*i as f64)?;
                if v.sigma <= 0. {
                    return Err(SaddleError::Invalid("zero interior tunneling momentum".into()));
                }
                sum += simpson_weight(i,n)*v.sigma*v.speed;
            }
            action += sum*h/3.;
        }
    }
    Ok(action)
}

/// Project +∇V onto the mass-weighted unit tangent at each image. Propagate
/// in spline arc length s; λ is only the parameter of the coordinate spline.
pub(in crate::lineb) fn force_clock<S: PointSurface>(
    surface: &S, masses: &Masses, curve: &Curve, energy: f64, pairs: usize,
) -> Result<(f64, TimeTable), SaddleError> {
    let mut arc = vec![0.];
    let mut lambda = vec![0.];
    let mut image_arc = vec![0.];
    let panels = 8 * pairs;
    let metric = |x| l2(curve.tangent(x).view());
    for chord in curve.knots.windows(2) {
        for i in 1..=panels {
            let a = *lambda.last().unwrap();
            let b = if i == panels {chord[1]} else {chord[0]+(chord[1]-chord[0])*i as f64/panels as f64};
            let ds = (b-a)*(metric(a)+4.*metric((a+b)/2.)+metric(b))/6.;
            if !ds.is_finite() || ds <= 0. {
                return Err(SaddleError::Invalid("invalid spline arc length".into()));
            }
            arc.push(arc.last().unwrap()+ds);
            lambda.push(b);
        }
        image_arc.push(*arc.last().unwrap());
    }
    let inverse = Spline::new(&arc, &lambda);
    let forces: Vec<_> = curve.knots.iter().map(|&x| {
        let sample = sample(surface,masses,curve,energy,x)?;
        Ok(sample.slope/sample.speed)
    }).collect::<Result<_,SaddleError>>()?;
    let last = forces.len()-1;
    if !(forces[0]>0. && forces[last]<0.) {
        return Err(SaddleError::Invalid("both image forces must point into the turning interval".into()));
    }
    let length = image_arc[last];
    let f = Spline::new(&image_arc, &forces);
    // Launch from the end whose integrated force reaches its return turn
    // inside the tabulated arc. The resulting turning coordinates are retained.
    let from_right = f.primitive(length)>0.;
    let sign = if from_right {-1.} else {1.};
    let mut s = if from_right {length} else {0.};
    let max_force = forces.iter().map(|x| x.abs()).fold(0., f64::max);
    let dt = (length/max_force).sqrt()/(64.*pairs as f64);
    let mut v = 0.;
    let mut a = f.value(s);
    let mut t = 0.;
    let mut knots = vec![TimeKnot { lambda: inverse.value(s), time: 0., velocity: 0. }];
    for _ in 0..1_000_000 {
        let pos = |h: f64| s+h*(v+0.5*h*a);
        let vel = |h: f64| v+0.5*h*(a+f.value(pos(h)));
        let turn = sign*vel(dt) <= 0.;
        let h = if turn {
            let (mut lo, mut hi) = (0., dt);
            for _ in 0..48 {
                let mid = 0.5*(lo+hi);
                if sign*vel(mid)>0. {lo=mid;} else {hi=mid;}
            }
            0.5*(lo+hi)
        } else {dt};
        let sn = pos(h);
        let vn = if turn {0.} else {vel(h)};
        // Verlet's return position can overstep an exact endpoint by O(dt²).
        let slack = dt*dt*max_force;
        if !sn.is_finite() || !vn.is_finite() || sn < -slack || sn > length+slack {
            return Err(SaddleError::Invalid("force-interpolated orbit leaves the image interval".into()));
        }
        t += h;
        let bounded = sn.clamp(0.,length);
        knots.push(TimeKnot{lambda:inverse.value(bounded),time:t,velocity:vn*inverse.derivative(bounded)});
        if turn {
            if from_right {
                knots.reverse();
                for knot in &mut knots {
                    knot.time=t-knot.time;
                    knot.velocity= -knot.velocity;
                }
            }
            return Ok((t,TimeTable {knots,linear:false}));
        }
        s=sn; v=vn; a=f.value(s);
    }
    Err(SaddleError::Invalid("force-interpolated orbit has no resolved return turn".into()))
}
