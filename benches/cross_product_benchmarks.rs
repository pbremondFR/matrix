#![feature(test)]
use criterion::{black_box, criterion_group, criterion_main, Criterion};

extern crate test;
use matrix::{funcs::cross_product, *};
use rand::distributions::{Distribution, Uniform};

fn criterion_benchmark(c: &mut Criterion) {
	let mut rng = rand::thread_rng();
	let die = Uniform::from(-200.0_f32..200.0_f32);

    c.bench_function("cross product", |b| {
		let u = Vector::from([die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)]);
		let v = Vector::from([die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)]);
		b.iter(|| {
			let n = black_box(42);
			cross_product(&u, &v)
		})
	});
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
