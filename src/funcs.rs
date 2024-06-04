use crate::{math_traits::*, vec_struct::*, Matrix};

pub fn linear_combination<K: Mathable, const N: usize>(u: &[Vector<N, K>], coefs: &[K]) -> Vector<N, K>
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
			res[j] = u[i][j].mul_add(coef, res[j]);
		}
	}
	return res;
}

pub fn cross_product<K: Mathable>(u: &Vector<3, K>, v: &Vector<3, K>) -> Vector<3, K> {

	// Vector::<3, K>::from([
	// 	u[1].mul_add(v[2], -u[2] * v[1]),	// u[1] * v[2] - u[2] * v[1],
	// 	u[2].mul_add(v[0], -u[0] * v[2]),	// u[2] * v[0] - u[0] * v[2],
	// 	u[0].mul_add(v[1], -u[1] * v[0])	// u[0] * v[1] - u[1] * v[0],
	// ])
	Vector::<3, K>::from([
		u[1] * v[2] - u[2] * v[1],
		u[2] * v[0] - u[0] * v[2],
		u[0] * v[1] - u[1] * v[0],
	])
}

pub fn projection(fov: f32, ratio: f32, near: f32, far: f32) -> Matrix::<4, 4, f32> {
	const DEG_TO_RAD: f32 = std::f32::consts::PI / 180.0;

	let fov = fov * ratio;
	let scaling_factor: f32 = 1.0 / f32::tan( (fov / 2.0) * DEG_TO_RAD );
	let yolo1 = -far / (far - near);
	let yolo2 = -(far * near) / (far - near);
	let projmat = Matrix::<4, 4, f32>::from([
		[scaling_factor, 0.0, 0.0, 0.0],
		[0.0, scaling_factor, 0.0, 0.0],
		[0.0, 0.0, yolo1, -1.0],
		[0.0, 0.0, yolo2, 0.0],
	]);
	return projmat;
}

#[cfg(test)]
mod tests {
	#[allow(unused_imports)]
	use crate::macros::assert_approx_eq;
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
	fn test_cross_product() {
		let u = Vector::from([0., 0., 1.]);
		let v = Vector::from([1., 0., 0.]);
		assert_eq!(cross_product(&u, &v), vector!(0.0, 1.0, 0.0));

		let u = Vector::from([1., 2., 3.]);
		let v = Vector::from([4., 5., 6.]);
		assert_eq!(cross_product(&u, &v), vector!(-3.0, 6.0, -3.0));

		let u = Vector::from([4., 2., -3.]);
		let v = Vector::from([-2., -5., 16.]);
		assert_eq!(cross_product(&u, &v), vector!(17.0, -58.0, -16.0));
	}

	#[test]
	fn projection_matrix() {
		let proj = projection(70.0, 1.0, 1.0, 50.0);
		println!("{}, {}, {}, {}", proj[0][0], proj[0][1], proj[0][2], proj[0][3]);
		println!("{}, {}, {}, {}", proj[1][0], proj[1][1], proj[1][2], proj[1][3]);
		println!("{}, {}, {}, {}", proj[2][0], proj[2][1], proj[2][2], proj[2][3]);
		println!("{}, {}, {}, {}", proj[3][0], proj[3][1], proj[3][2], proj[3][3]);
	}
}
