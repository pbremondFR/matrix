#![allow(incomplete_features)]
#![feature(min_specialization)]
#![feature(associated_type_defaults)]
#![feature(generic_const_exprs)]
#![feature(test)]
pub mod math_traits;
pub mod mat_struct;
pub mod vec_struct;
pub mod funcs;
pub mod macros;

pub use vec_struct::*;
pub use mat_struct::*;
pub use math_traits::*;
