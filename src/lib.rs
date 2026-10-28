pub mod error;
pub mod model;
pub mod node;
pub mod nodes;
pub mod sample;
pub mod tensor;

pub mod prelude {
    pub use crate::model::Model;
    pub use crate::node::Support;
    pub use crate::nodes::{Function, Potential};
}
