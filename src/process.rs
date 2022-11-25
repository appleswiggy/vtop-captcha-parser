use image::io::Reader;
use serde_json::Value;

use std::cmp::{max, min};
use std::io::Cursor;

use crate::weights::DATA;

const HEIGHT: usize = 40;
const WIDTH: usize = 200;
const CAPTCHA_CHARS: &str = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

pub struct ImageProcessor {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

impl ImageProcessor {
    pub fn new() -> Self {
        // unwrap() is safe because parsing the hardcoded JSON data never panics.
        let json: Value = serde_json::from_str(DATA).unwrap();

        let weights_value = json["weights"].clone();
        let biases_value = json["biases"].clone();

        let weights: Vec<Vec<f64>> = serde_json::from_value(weights_value).unwrap();
        let biases: Vec<f64> = serde_json::from_value(biases_value).unwrap();

        ImageProcessor { weights, biases }
    }

    pub fn process(&self, byte_array: &Vec<u8>) -> Result<String, Box<dyn std::error::Error>> {
        let mut reader = Reader::new(Cursor::new(byte_array));
        reader.set_format(image::ImageFormat::Jpeg);

        let pixels = reader.decode()?.to_rgba8().into_raw();

        Ok(self.process_pixels(&pixels)?)
    }

    fn process_pixels(&self, pixels: &Vec<u8>) -> Result<String, Box<dyn std::error::Error>> {
        let sat = saturate(&pixels);
        let def = de_flatten(&sat);
        let block_list = get_blocks(&def);

        let mut captcha_text = String::new();

        for block in block_list.iter() {
            let processed: Vec<Vec<u8>> = pre_process(block);
            let flattened: Vec<Vec<u8>> = [flatten(&processed)].to_vec();

            let multiplied: Vec<Vec<f64>> = mat_multiply(&flattened, &self.weights);
            let added: Vec<f64> = mat_add(multiplied.get(0).unwrap(), &self.biases);

            let arr: Vec<f64> = softmax(&added);
            let index_of_max: usize = arr
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap();

            captcha_text += CAPTCHA_CHARS.get(index_of_max..index_of_max + 1).unwrap();
        }

        Ok(captcha_text)
    }
}

fn saturate(pixels: &Vec<u8>) -> Vec<u8> {
    let mut sat: Vec<u8> = Vec::with_capacity(pixels.len() / 4);

    let mut index: usize = 0;
    while index < pixels.len() {
        let min_value = min(
            pixels.get(index).unwrap(),
            min(
                pixels.get(index + 1).unwrap(),
                pixels.get(index + 2).unwrap(),
            ),
        );
        let max_value = max(
            pixels.get(index).unwrap(),
            max(
                pixels.get(index + 1).unwrap(),
                pixels.get(index + 2).unwrap(),
            ),
        );

        sat.push((((*max_value as usize - *min_value as usize) * 255) / *max_value as usize) as u8);
        index += 4;
    }
    return sat;
}

fn flatten(block: &Vec<Vec<u8>>) -> Vec<u8> {
    let mut flattened: Vec<u8> = Vec::with_capacity(block.len() * block.get(0).unwrap().len());
    for i in 0..block.len() {
        for j in 0..block.get(0).unwrap().len() {
            flattened.push(*(block.get(i).unwrap().get(j).unwrap()));
        }
    }
    return flattened;
}

fn de_flatten(saturated: &Vec<u8>) -> Vec<Vec<u8>> {
    let mut de_flattened: Vec<Vec<u8>> = Vec::with_capacity(HEIGHT);

    for i in 0..HEIGHT {
        let mut arr: Vec<u8> = Vec::with_capacity(WIDTH);
        for j in 0..WIDTH {
            arr.push(*(saturated.get(i * WIDTH + j).unwrap()));
        }
        de_flattened.push(arr);
    }

    return de_flattened;
}

fn get_blocks(deflatted: &Vec<Vec<u8>>) -> Vec<Vec<Vec<u8>>> {
    let mut blocks_list: Vec<Vec<Vec<u8>>> = Vec::with_capacity(6);

    let mut a: usize = 0;

    while a < 6 {
        let x1 = (a + 1) * 25 + 2;
        let y1 = 7 + 5 * (a % 2) + 1;

        let x2 = (a + 2) * 25 + 1;
        let y2 = 35 - 5 * ((a + 1) % 2);

        blocks_list.push(
            deflatted[y1..y2]
                .to_vec()
                .iter()
                .map(|s| s[x1..x2].to_vec())
                .collect(),
        );
        a += 1;
    }

    return blocks_list;
}

fn pre_process(block: &Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let mut avg: f64 = 0.0;

    for i in block.iter() {
        for j in i.iter() {
            avg += *j as f64;
        }
    }

    avg = avg / (24 * 22) as f64;
    let mut processed: Vec<Vec<u8>> = Vec::with_capacity(block.len());

    for i in 0..block.len() {
        let len = block.get(0).unwrap().len();
        let mut arr: Vec<u8> = Vec::with_capacity(len);

        for j in 0..len {
            if *(block.get(i).unwrap().get(j).unwrap()) > (avg as u8) {
                arr.push(1);
            } else {
                arr.push(0);
            }
        }
        processed.push(arr);
    }

    return processed;
}

fn mat_multiply(matrix: &Vec<Vec<u8>>, weights: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let x = matrix.len();
    let z = matrix.get(0).unwrap().len();
    let y = weights.get(0).unwrap().len();

    assert!(weights.len() == z);

    let product_row: Vec<f64> = vec![0.0; y];
    let mut product: Vec<Vec<f64>> = vec![product_row; x];

    for i in 0..x {
        for j in 0..y {
            for k in 0..z {
                product[i][j] += matrix[i][k] as f64 * weights[k][j];
            }
        }
    }

    return product;
}

fn mat_add(first: &Vec<f64>, second: &Vec<f64>) -> Vec<f64> {
    let len = first.len();
    let mut arr: Vec<f64> = Vec::with_capacity(len);

    for i in 0..len {
        arr.push(*first.get(i).unwrap() + *second.get(i).unwrap());
    }
    return arr;
}

fn softmax(arg: &Vec<f64>) -> Vec<f64> {
    let mut n_arr = arg.clone();
    let mut s: f64 = 0.0;

    for i in n_arr.iter() {
        s += i.exp();
    }

    for i in 0..arg.len() {
        n_arr.push((arg.get(i).unwrap().exp()) / s);
    }

    return n_arr;
}
