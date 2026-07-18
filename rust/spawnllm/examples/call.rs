use spawnllm::{CallOpts, blocking};

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Capital {
    city: String,
}

fn main() {
    let text = blocking::call("Reply with exactly: pong", CallOpts::default()).expect("call");
    println!("call: {text}");
    let capital: Capital =
        blocking::extract("What is the capital of France?", CallOpts::default()).expect("extract");
    println!("extract: {}", capital.city);
}
