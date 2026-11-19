//! Rust `to_abi` ordinals. The eight Python IntEnum tables are listed
//! first. `FreeEndSide` is the C discriminant (`0` for `First`, `1` for
//! `Last`); Python takes the strings `first` and `last`.

use rgsaddle::{
    CellChart, EigenDevice, ExpandKind, ForceGate, FreeEndSide, HessUpdate, IrcKind, MinModeKind,
    Reparametrization, RestrictedKind,
};

#[test]
fn to_abi_matches_the_python_and_c_table() {
    assert_eq!(ForceGate::L2Norm.to_abi(), 0);
    assert_eq!(ForceGate::LinfNorm.to_abi(), 1);
    assert_eq!(ForceGate::MaxForceOnAtom.to_abi(), 2);

    assert_eq!(HessUpdate::Bfgs.to_abi(), 0);
    assert_eq!(HessUpdate::TsBfgs.to_abi(), 1);

    assert_eq!(IrcKind::Gs2.to_abi(), 0);
    assert_eq!(IrcKind::Morokuma.to_abi(), 1);
    assert_eq!(IrcKind::SellaSphere.to_abi(), 2);

    assert_eq!(MinModeKind::Dimer.to_abi(), 0);
    assert_eq!(MinModeKind::Lanczos.to_abi(), 1);

    assert_eq!(EigenDevice::Host.to_abi(), 0);
    assert_eq!(EigenDevice::Dlpk.to_abi(), 1);

    assert_eq!(ExpandKind::Lanczos.to_abi(), 0);
    assert_eq!(ExpandKind::Gd.to_abi(), 1);
    assert_eq!(ExpandKind::Jd0.to_abi(), 2);
    assert_eq!(ExpandKind::Jd0Alt.to_abi(), 3);
    assert_eq!(ExpandKind::Mjd0.to_abi(), 4);
    assert_eq!(ExpandKind::Mjd0Alt.to_abi(), 5);

    assert_eq!(CellChart::Entries.to_abi(), 0);
    assert_eq!(CellChart::LogDeform.to_abi(), 1);

    assert_eq!(RestrictedKind::TrustRegion.to_abi(), 0);
    assert_eq!(RestrictedKind::MaxInternalStep.to_abi(), 1);
    assert_eq!(RestrictedKind::RestrictedAtomicStep.to_abi(), 2);

    assert_eq!(Reparametrization::Linear.to_abi(), 0);
    assert_eq!(Reparametrization::NotAKnot.to_abi(), 1);
    for code in 0..2 {
        assert_eq!(
            Reparametrization::from_abi(code).map(Reparametrization::to_abi),
            Some(code)
        );
    }
    assert_eq!(Reparametrization::from_abi(2), None);

    assert_eq!(FreeEndSide::First.to_abi(), 0);
    assert_eq!(FreeEndSide::Last.to_abi(), 1);
    for code in 0..2 {
        assert_eq!(
            FreeEndSide::from_abi(code).map(FreeEndSide::to_abi),
            Some(code)
        );
    }
    assert_eq!(FreeEndSide::from_abi(2), None);
    assert_eq!(FreeEndSide::from_abi(-1), None);
}
