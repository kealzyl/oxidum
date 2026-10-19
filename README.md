# Kealzyl Oxidum (Rust)

Educational next-token language model implemented in Rust.

## Requirements
Install Rust and Cargo from https://rustup.rs/

## Run
Place the training text in `INSTRUCTIONS.md` in this folder, then run:

```bash
cargo run --release
```

Cargo downloads the required crates on first build. The default settings use 1,500 epochs; edit the constants at the top of `src/main.rs` to change them.

## Commands
- `/temperature 0.8` — change sampling temperature (0.05–2.0)
- `/export [file.kpz]` — save the model
- `/quit` — exit chat

Model files use Rust's `.kpz` format and are not compatible with the original Python `.npz` model files.
