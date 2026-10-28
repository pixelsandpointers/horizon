# horizon

> This is alpha software. Please be aware that the API will change and may break things from time to time.

Install via crates.io: `cargo add horizon`.

NUTS sampling (+ RxInfer.jl message passing) on factor graphs, with analytic gradients for built-in nodes, and [burn](https://burn.dev)'s autodiff for nodes written with tensors. Currently AD doesn't work in Python.

```rust
use horizon::prelude::*;

let mut m = Model::new();
let mu = m.random("mu");
m.normal_mean_variance(mu, 0.0, 100.0);
let y = m.data("y", [4.9, 5.3, 4.6, 5.1]);
m.normal_mean_variance(y, mu, 1.0);
println!("{}", m.sample().run()?.summary());
```

```sh
cargo run --release --example eight_schools
cargo test --release
```

## Python

Bindings live in `python/`, built with [maturin](https://www.maturin.rs), with type stubs in `python/horizon.pyi`. Potentials and functions are Python callables over scalars.

```python
import horizon as hz

m = hz.Model()
mu = m.random("mu")
m.normal_mean_variance(mu, 0.0, 100.0)
y = m.data("y", [4.9, 5.3, 4.6, 5.1])
m.normal_mean_variance(y, mu, 1.0)
print(m.sample().summary())
```

```sh
cd python
uv sync                 # builds the extension into .venv
uv run pytest
uv run python examples/eight_schools.py
```

## Acknowledgments
The crate name was owned before by [Leonard Lesinski](https://github.com/Le0X8).
