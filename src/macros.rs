macro_rules! assert_approx_eq {
    ($left:expr, $right:expr, $epsilon:expr) => {{
        let (left, right, epsilon) = ($left, $right, $epsilon);
        assert!(
            (left - right).abs() < epsilon,
            "{left} is not equal to {right} account error margin of {epsilon}"
        );
    }};
}

pub(crate) use assert_approx_eq;
