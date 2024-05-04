macro_rules! assert_approx_eq {
    ($left:expr, $right:expr, $epsilon:expr) => {{
        let (left, right, epsilon) = ($left, $right, $epsilon);
        assert!(
            (left - right).abs() < epsilon,
            "{left} != {right} accounting error margin of {epsilon}"
        );
    }};
}

pub(crate) use assert_approx_eq;
