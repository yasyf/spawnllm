fn main() {
    println!("cargo::rerun-if-env-changed=SPAWNLLM_CORE_SRC_HASH");
}
