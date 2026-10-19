use ndarray::{Array1, Array2, Axis};
use rand::{distributions::WeightedIndex, prelude::*, rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

const README_PATH: &str = "INSTRUCTIONS.md";
const DEFAULT_MODEL_PATH: &str = "kealzyl_oxidum.kpz";
const CONTEXT_SIZE: usize = 3;
const EMBED_DIM: usize = 24;
const HIDDEN_SIZE: usize = 64;
const EPOCHS: usize = 1500;
const LEARNING_RATE: f32 = 0.01;
const BATCH_SIZE: usize = 32;
const SEED: u64 = 42;

#[derive(Serialize, Deserialize, Clone)]
struct Model {
    vocabulary: Vec<String>,
    e: Array2<f32>,
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

fn prompt(message: &str) -> String {
    print!("{message}");
    let _ = io::stdout().flush();
    let mut value = String::new();
    if io::stdin().read_line(&mut value).is_err() {
        return String::new();
    }
    value.trim().to_string()
}

fn absolute(path: &str) -> String {
    fs::canonicalize(path)
        .unwrap_or_else(|_| PathBuf::from(path))
        .display()
        .to_string()
}

fn tokenize(text: &str) -> Vec<String> {
    let re = Regex::new(r"\w+|[^\w\s]").expect("valid token regex");
    re.find_iter(&text.to_lowercase())
        .map(|m| m.as_str().to_string())
        .collect()
}

fn softmax(logits: &Array2<f32>) -> Array2<f32> {
    let mut output = logits.clone();
    for mut row in output.rows_mut() {
        let max_value = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for value in row.iter_mut() {
            *value = (*value - max_value).clamp(-60.0, 60.0).exp();
            sum += *value;
        }
        let denominator = sum.max(1e-12);
        for value in row.iter_mut() {
            *value /= denominator;
        }
    }
    output
}

fn make_dataset(ids: &[usize]) -> (Vec<Vec<usize>>, Vec<usize>) {
    let mut inputs = Vec::with_capacity(ids.len().saturating_sub(CONTEXT_SIZE));
    let mut targets = Vec::with_capacity(ids.len().saturating_sub(CONTEXT_SIZE));

    for index in CONTEXT_SIZE..ids.len() {
        inputs.push(ids[index - CONTEXT_SIZE..index].to_vec());
        targets.push(ids[index]);
    }
    (inputs, targets)
}

fn random_matrix(
    rng: &mut StdRng,
    rows: usize,
    columns: usize,
    scale: f32,
) -> Array2<f32> {
    Array2::from_shape_fn((rows, columns), |_| {
        let value: f32 = StandardNormal.sample(rng);
        value * scale
    })
}

fn initialize_model(vocabulary: Vec<String>) -> Model {
    let vocab_size = vocabulary.len();
    let mut rng = StdRng::seed_from_u64(SEED);

    Model {
        vocabulary,
        e: random_matrix(&mut rng, vocab_size, EMBED_DIM, 0.02),
        w1: random_matrix(
            &mut rng,
            CONTEXT_SIZE * EMBED_DIM,
            HIDDEN_SIZE,
            (2.0 / (CONTEXT_SIZE * EMBED_DIM) as f32).sqrt(),
        ),
        b1: Array1::zeros(HIDDEN_SIZE),
        w2: random_matrix(
            &mut rng,
            HIDDEN_SIZE,
            vocab_size,
            (1.0 / HIDDEN_SIZE as f32).sqrt(),
        ),
        b2: Array1::zeros(vocab_size),
    }
}

fn forward(
    model: &Model,
    contexts: &[Vec<usize>],
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let batch_size = contexts.len();
    let mut flattened = Array2::<f32>::zeros((batch_size, CONTEXT_SIZE * EMBED_DIM));

    for (batch_index, context) in contexts.iter().enumerate() {
        for (position, &token_id) in context.iter().enumerate() {
            for dimension in 0..EMBED_DIM {
                flattened[[batch_index, position * EMBED_DIM + dimension]] =
                    model.e[[token_id, dimension]];
            }
        }
    }

    let mut pre_activation = flattened.dot(&model.w1);
    for mut row in pre_activation.rows_mut() {
        row += &model.b1;
    }

    let hidden = pre_activation.mapv(|value| value.max(0.0));
    let mut logits = hidden.dot(&model.w2);
    for mut row in logits.rows_mut() {
        row += &model.b2;
    }

    (logits, flattened, pre_activation, hidden)
}

fn train(readme_path: &str) -> Option<Model> {
    let mut path = PathBuf::from(readme_path);

    if !path.is_file() {
        println!("\nCould not find {:?}.", readme_path);
        println!("Put README.md beside the program or enter another path.");
        let entered_path = prompt("README path (or press Enter to cancel): ");
        if entered_path.is_empty() {
            return None;
        }
        path = PathBuf::from(entered_path);
    }

    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("Could not read file: {error}");
            return None;
        }
    };

    let tokens = tokenize(&text);
    if tokens.len() <= CONTEXT_SIZE {
        println!("The README does not contain enough text to train.");
        return None;
    }

    let mut counts: HashMap<String, usize> = HashMap::new();
    for token in &tokens {
        *counts.entry(token.clone()).or_insert(0) += 1;
    }

    // Stable vocabulary order: most frequent first, alphabetical for ties.
    let mut vocabulary: Vec<String> = counts.keys().cloned().collect();
    vocabulary.sort_by(|left, right| {
        counts[right]
            .cmp(&counts[left])
            .then_with(|| left.cmp(right))
    });

    let token_to_id: HashMap<String, usize> = vocabulary
        .iter()
        .enumerate()
        .map(|(index, token)| (token.clone(), index))
        .collect();

    let ids: Vec<usize> = tokens.iter().map(|token| token_to_id[token]).collect();
    let (x, y) = make_dataset(&ids);
    let vocab_size = vocabulary.len();

    println!("\nFile: {}", path.display());
    println!("Tokens: {}", tokens.len());
    println!("Vocabulary: {vocab_size}");
    println!("Training examples: {}", x.len());
    println!("Epochs: {EPOCHS}");
    println!("Initializing model...");

    let mut model = initialize_model(vocabulary);
    let mut rng = StdRng::seed_from_u64(SEED);

    for epoch in 0..EPOCHS {
        let mut order: Vec<usize> = (0..x.len()).collect();
        order.shuffle(&mut rng);

        let mut total_loss = 0.0_f64;
        let mut batches = 0_usize;

        for indices in order.chunks(BATCH_SIZE) {
            let xb: Vec<Vec<usize>> = indices.iter().map(|&index| x[index].clone()).collect();
            let yb: Vec<usize> = indices.iter().map(|&index| y[index]).collect();
            let batch_count = xb.len();

            let (logits, flattened, pre_activation, hidden) = forward(&model, &xb);
            let mut probabilities = softmax(&logits);

            for row in 0..batch_count {
                total_loss += -(probabilities[[row, yb[row]]].max(1e-12).ln() as f64)
                    / batch_count as f64;
            }
            batches += 1;

            // Gradient of cross-entropy loss combined with softmax.
            for row in 0..batch_count {
                probabilities[[row, yb[row]]] -= 1.0;
            }
            probabilities.mapv_inplace(|value| value / batch_count as f32);

            let dw2 = hidden.t().dot(&probabilities);
            let db2 = probabilities.sum_axis(Axis(0));
            let dhidden = probabilities.dot(&model.w2.t());

            let mut dpre = dhidden;
            for row in 0..dpre.nrows() {
                for column in 0..dpre.ncols() {
                    if pre_activation[[row, column]] <= 0.0 {
                        dpre[[row, column]] = 0.0;
                    }
                }
            }

            let dw1 = flattened.t().dot(&dpre);
            let db1 = dpre.sum_axis(Axis(0));
            let dflattened = dpre.dot(&model.w1.t());

            // Accumulate gradients for repeated token IDs.
            let mut de = Array2::<f32>::zeros(model.e.raw_dim());
            for (row, context) in xb.iter().enumerate() {
                for (position, &token_id) in context.iter().enumerate() {
                    for dimension in 0..EMBED_DIM {
                        de[[token_id, dimension]] +=
                            dflattened[[row, position * EMBED_DIM + dimension]];
                    }
                }
            }

            // Gradient clipping.
            for (parameter, gradient) in model.w2.iter_mut().zip(dw2.iter()) {
                *parameter -= LEARNING_RATE * gradient.clamp(-1.0, 1.0);
            }
            for (parameter, gradient) in model.b2.iter_mut().zip(db2.iter()) {
                *parameter -= LEARNING_RATE * gradient.clamp(-1.0, 1.0);
            }
            for (parameter, gradient) in model.w1.iter_mut().zip(dw1.iter()) {
                *parameter -= LEARNING_RATE * gradient.clamp(-1.0, 1.0);
            }
            for (parameter, gradient) in model.b1.iter_mut().zip(db1.iter()) {
                *parameter -= LEARNING_RATE * gradient.clamp(-1.0, 1.0);
            }
            for (parameter, gradient) in model.e.iter_mut().zip(de.iter()) {
                *parameter -= LEARNING_RATE * gradient.clamp(-1.0, 1.0);
            }
        }

        if epoch == 0 || (epoch + 1) % 10 == 0 || epoch + 1 == EPOCHS {
            println!(
                "Epoch {:3}/{EPOCHS} | loss: {:.4}",
                epoch + 1,
                total_loss / batches.max(1) as f64
            );
        }
    }

    println!("\nTraining complete.");
    Some(model)
}

fn save_model(model: &Model, path: &str) -> bool {
    match bincode::serialize(model) {
        Ok(bytes) => match fs::write(path, bytes) {
            Ok(()) => {
                println!("Model saved to: {}", absolute(path));
                true
            }
            Err(error) => {
                eprintln!("Could not save model: {error}");
                false
            }
        },
        Err(error) => {
            eprintln!("Could not serialize model: {error}");
            false
        }
    }
}

fn load_model(path: &str) -> Option<Model> {
    if !Path::new(path).is_file() {
        println!("Model file not found: {path}");
        return None;
    }

    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("Could not read model: {error}");
            return None;
        }
    };

    let model: Model = match bincode::deserialize(&bytes) {
        Ok(model) => model,
        Err(error) => {
            eprintln!("Could not load model: {error}");
            return None;
        }
    };

    let vocab_size = model.vocabulary.len();
    if model.e.dim() != (vocab_size, EMBED_DIM)
        || model.w1.dim() != (CONTEXT_SIZE * EMBED_DIM, HIDDEN_SIZE)
        || model.b1.len() != HIDDEN_SIZE
        || model.w2.dim() != (HIDDEN_SIZE, vocab_size)
        || model.b2.len() != vocab_size
    {
        println!("Invalid model file: parameter shapes do not match.");
        return None;
    }

    println!("Loaded model: {}", absolute(path));
    println!("Vocabulary: {vocab_size}");
    Some(model)
}

fn detokenize(tokens: &[String]) -> String {
    let mut text = tokens.join(" ");
    text = Regex::new(r"\s+([.,!?;:%)\]}])")
        .unwrap()
        .replace_all(&text, "$1")
        .to_string();
    text = Regex::new(r"([(\[{])\s+")
        .unwrap()
        .replace_all(&text, "$1")
        .to_string();
    Regex::new(r"\s+'\s+")
        .unwrap()
        .replace_all(&text, "'")
        .to_string()
}

fn generate(model: &Model, prompt_text: &str, max_new_tokens: usize, temperature: f32) -> String {
    let token_to_id: HashMap<&str, usize> = model
        .vocabulary
        .iter()
        .enumerate()
        .map(|(index, token)| (token.as_str(), index))
        .collect();

    let mut ids: Vec<usize> = tokenize(prompt_text)
        .iter()
        .filter_map(|token| token_to_id.get(token.as_str()).copied())
        .collect();

    if ids.is_empty() {
        ids.push(0);
    }

    let mut rng = rand::thread_rng();
    let temperature = temperature.max(0.05);

    for _ in 0..max_new_tokens {
        let start = ids.len().saturating_sub(CONTEXT_SIZE);
        let mut context = ids[start..].to_vec();

        if context.len() < CONTEXT_SIZE {
            let mut padded = vec![0; CONTEXT_SIZE - context.len()];
            padded.extend(context);
            context = padded;
        }

        let (logits, _, _, _) = forward(model, &[context]);
        let scaled_logits = logits.mapv(|value| value / temperature);
        let probabilities = softmax(&scaled_logits);
        let weights: Vec<f64> = probabilities.row(0).iter().map(|&v| v as f64).collect();

        let next_id = match WeightedIndex::new(&weights) {
            Ok(distribution) => distribution.sample(&mut rng),
            Err(_) => rng.gen_range(0..model.vocabulary.len()),
        };
        ids.push(next_id);
    }

    let output_tokens: Vec<String> = ids
        .iter()
        .map(|&index| model.vocabulary[index].clone())
        .collect();
    detokenize(&output_tokens)
}

fn chat(model: &Model) {
    let mut temperature = 0.8_f32;

    println!("\nKealzyl Oxidum is ready.");
    println!("Type a prompt to generate text.");
    println!("Commands: /export [file.kpz], /temperature 0.8, /quit");

    loop {
        let prompt_text = prompt("\nYou: ");
        if prompt_text.is_empty() {
            continue;
        }

        let lower = prompt_text.to_lowercase();

        if ["/quit", "/exit", "quit", "exit"].contains(&lower.as_str()) {
            println!("Goodbye.");
            break;
        }

        if lower.starts_with("/export") {
            let mut parts = prompt_text.splitn(2, char::is_whitespace);
            parts.next();
            let mut path = parts.next().unwrap_or("").trim().to_string();

            if path.is_empty() {
                path = DEFAULT_MODEL_PATH.to_string();
            }
            if !path.to_lowercase().ends_with(".kpz") {
                path.push_str(".kpz");
            }

            save_model(model, &path);
            continue;
        }

        if lower.starts_with("/temperature") {
            let parts: Vec<&str> = prompt_text.split_whitespace().collect();
            if parts.len() == 2 {
                match parts[1].parse::<f32>() {
                    Ok(value) => {
                        temperature = value.clamp(0.05, 2.0);
                        println!("Temperature set to {temperature:.2}");
                    }
                    Err(_) => println!("Usage: /temperature 0.8"),
                }
            } else {
                println!("Current temperature: {temperature:.2}");
            }
            continue;
        }

        print!("\nKealzyl Oxidum: ");
        let _ = io::stdout().flush();
        println!("{}", generate(model, &prompt_text, 60, temperature));
    }
}

fn main() {
    println!("==========================================");
    println!("          KEALZYL OXIDUM");
    println!("   Tiny educational language model");
    println!("==========================================");
    println!("1. Train from README.md");
    println!("2. Load a saved model");
    println!("3. Exit");

    loop {
        let choice = prompt("\nChoose an option (1-3): ");

        match choice.as_str() {
            "1" => {
                if let Some(model) = train(README_PATH) {
                    save_model(&model, DEFAULT_MODEL_PATH);
                    chat(&model);
                }
                return;
            }
            "2" => {
                let entered_path = prompt(&format!("Model path [{DEFAULT_MODEL_PATH}]: "));
                let path = if entered_path.is_empty() {
                    DEFAULT_MODEL_PATH
                } else {
                    entered_path.as_str()
                };
                if let Some(model) = load_model(path) {
                    chat(&model);
                }
                return;
            }
            "3" => {
                println!("Goodbye.");
                return;
            }
            _ => println!("Please enter 1, 2, or 3."),
        }
    }
}
