use std::path::Path;
use std::fs::read;

use process::ImageProcessor;

mod process;
mod weights;

pub struct Parser {
    processor: ImageProcessor,
}

impl Parser {
    pub fn new() -> Self {
        Parser {
            processor: ImageProcessor::new(),
        }
    }

    pub fn parse_from_file<P: AsRef<Path>>(&self, path: P) -> Result<String, Box<dyn std::error::Error>> {
        let byte_array: Vec<u8> = read(path)?;
        self.processor.process(&byte_array)
    }

    pub fn parse_from_base64(&self, b64_data: &str) -> Result<String, Box<dyn std::error::Error>> {
        let byte_array: Vec<u8> = base64::decode(b64_data)?;
        self.processor.process(&byte_array)
    }

    pub fn parse_from_bytes(&self, byte_array: &Vec<u8>) -> Result<String, Box<dyn std::error::Error>> {
        self.processor.process(&byte_array)
    }
}
