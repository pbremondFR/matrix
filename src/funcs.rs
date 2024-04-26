use crate::{math_traits::Mathable, vec_struct::*};

fn linear_combination<K: Mathable, const N: usize>(u: &[Vector<N, K>], coefs: &[K]) -> Vector<N, K> {
	if coefs.len() != u.len() {
		panic!();
	}
	let mut res = Vector::<N, K>::new();
	for i in 0..u.len() {
		res = res + u[i] * coefs[i];
	}
	return res;
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
}
