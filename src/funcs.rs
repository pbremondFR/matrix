use crate::{math_traits::{Mathable, Norm, RealReal}, vec_struct::*};

fn linear_combination<K: Mathable, const N: usize>(u: &[Vector<N, K>], coefs: &[K]) -> Vector<N, K>
{
	if coefs.len() != u.len() {
		panic!();
	}
	// TODO: Benchmark me VS mul_add!

	// {
	// 	let mut res = Vector::<N, K>::new();
	// 	for i in 0..u.len() {
	// 		res = res + u[i] * coefs[i];
	// 	}
	// }
	let mut res = Vector::<N, K>::new();
	for i in 0..u.len() {
		let coef = coefs[i];
		for j in 0..N {
			let x = u[i][j];
			res[j] = x.mul_add(coef, res[j]);
		}
	}
	return res;
}

// Can't have this work out nicely for f32 and f64...
// A few options:
// 1. Make a generic AngleCos trait and derive it vectors of f32 and f64 and so on
// 2. Give up on that whole f32/f64/FixedPoint nonsense. This is a relatively small
// project, and everything that I'd be able to do in C++ here, Rust is not letting me do.
// Picking f32 as the "only" real number is allowed, so...
fn angle_cos<K, const N: usize>(u: &Vector<N, K>, v: &Vector<N, K>) -> impl RealReal + Mathable
where
	K: Mathable + RealReal
{
	u.dot(v) / (u.norm() * v.norm())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_1() {
		let e1 = Vector::from([1., 0., 0.]);
		let e2 = Vector::from([0., 1., 0.]);
		let e3 = Vector::from([0., 0., 1.]);

		let v1 = Vector::from([1., 2., 3.]);
		let v2 = Vector::from([0., 10., -100.]);

		let expected = vector!(10., -2., 0.5);
		let res = linear_combination(&[e1, e2, e3], &[10., -2., 0.5]);
		println!("{res}");
		assert_eq!(res, expected);

		let expected = vector!(10., 0., 230.);
		let res = linear_combination(&[v1, v2], &[10., -2.]);
		assert_eq!(res, expected);
	}

	#[test]
	#[should_panic]
	fn test_panic() {
		let e1 = Vector::from([1., 0., 0.]);
		let e2 = Vector::from([0., 1., 0.]);
		let e3 = Vector::from([0., 0., 1.]);
		let _ = linear_combination(&[e1, e2, e3], &[10., -2.]);
	}

	#[test]
	fn test_angle_cos() {
		let u = Vector::from([1., 0.]);
		let v = Vector::from([1., 0.]);
		// assert_eq!(angle_cos(&u, &v), 1.0);
		let foo = angle_cos(&u, &v);
		println!("TESTESTETS: {foo} - 1.0");

		let u = Vector::from([1., 0.]);
		let v = Vector::from([0., 1.]);
		// assert_eq!(angle_cos(&u, &v), 0.0);

		let u = Vector::<2, f64>::from([-1., 1.]);
		let v = Vector::<2, f64>::from([ 1., -1.]);
		// FUCK
		// assert_eq!(angle_cos(&u, &v), -1.0);

		let u = Vector::from([2., 1.]);
		let v = Vector::from([4., 2.]);
		// assert_eq!(angle_cos(&u, &v), 1.0);

		let u = Vector::from([1., 2., 3.]);
		let v = Vector::from([4., 5., 6.]);
		// assert_eq!(angle_cos(&u, &v), 0.974631846);

	}
}
