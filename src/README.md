# horizon

NUTS sampling (+ RxInfer.jl message passing) on factor graphs, with gradients from [burn](https://burn.dev)'s autodiff. 

```rust
use horizon::*;

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
