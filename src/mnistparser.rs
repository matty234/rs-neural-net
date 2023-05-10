use std::{fs::File, io::Read};
use byteorder::{ReadBytesExt, BigEndian};
use flate2::read::GzDecoder;
use ndarray::prelude::*;
use std::{io::Cursor};

#[derive(Clone)]
pub struct MnistImage {
    pub label: u8,
    pub pixels: Vec<u8>,
}

impl MnistImage {

    pub fn parse_from_bytes(label:u8, pixels: Vec<u8>) -> MnistImage {
        MnistImage {
            label,
            pixels,
        }
    }

    pub fn get_label(&self) -> u8 {
        self.label
    }

    pub fn write_to_bitmap(&self, filename: &str) {
        let mut file = File::create(filename).unwrap();
        let mut encoder = png::Encoder::new(&mut file, 28, 28);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&self.pixels).unwrap();

        writer.finish().unwrap();
    }

    pub fn get_f64_pixels(&self) -> Array2<f64> {
        let mut pixels = Array2::<f64>::zeros((28 * 28, 1));

        for (i, pixel) in self.pixels.iter().enumerate() {
            pixels[[i, 0]] = *pixel as f64;
        }
        pixels
    }


    pub fn one_hot_label(&self) -> Array2<f64> {
        let mut label = Array2::<f64>::zeros((10, 1));
        label[[self.label as usize, 0]] = 1.0;
        label
    }

    
}

pub struct MnistParser {
    pub images: Vec<MnistImage>,
}

impl MnistParser {
    pub fn new() -> MnistParser {
        MnistParser {
            images: Vec::new(),
        }
    }

    pub fn from_mnist_parser_get_range(&self, start: usize, end: usize) -> MnistParser {
        let mut new_parser = MnistParser::new();
        for i in start..end {
            new_parser.images.push(self.images[i].clone());
        }
        new_parser
    }



    pub fn parse(&mut self, image_file: File, label_file : File) {
     
        let mut decoder = GzDecoder::new(image_file);
        let mut decoder_labels = GzDecoder::new(label_file);
    
        let mut buffer = Vec::new();
        decoder.read_to_end(&mut buffer).unwrap();
    
        let mut buffer_labels = Vec::new();
        decoder_labels.read_to_end(&mut buffer_labels).unwrap();
    
        // parse bytes
        let mut cursor = Cursor::new(buffer);
        let mut cursor_labels = Cursor::new(buffer_labels);
    
        let magic_number = cursor.read_u32::<BigEndian>().unwrap();
        if magic_number != 2051 {
            panic!("Invalid magic number for cursor. Expected 2051, got {}", magic_number);
        }
    
        let magic_number_labels = cursor_labels.read_u32::<BigEndian>().unwrap();
        if magic_number_labels != 2049 {
            panic!("Invalid magic number for labels. Expected 2049, got {}", magic_number_labels);
        }
    
        let num_images = cursor.read_u32::<BigEndian>().unwrap();
        let num_images_labels = cursor_labels.read_u32::<BigEndian>().unwrap();
    
        println!("Number of images: {}", num_images);
    
        if num_images != num_images_labels {
            panic!("Number of images and labels do not match. Expected {}, got {}", num_images, num_images_labels);
        }
    
        let num_rows = cursor.read_u32::<BigEndian>().unwrap();
        let num_cols = cursor.read_u32::<BigEndian>().unwrap();
    
        println!("Number of rows: {}", num_rows);
        println!("Number of cols: {}", num_cols);

        for _ in 0..num_images {
            let label = cursor_labels.read_u8().unwrap();
            let mut pixels = Vec::new();
            for _ in 0..num_rows * num_cols {
                pixels.push(cursor.read_u8().unwrap());
            }
            self.images.push(MnistImage::parse_from_bytes(label, pixels));


        }
    }

    pub fn get_image_matrix(&self) -> Array2<f64> {
        let mut image_matrix = Array2::<f64>::zeros((28 * 28, self.images.len()));
        for (i, image) in self.images.iter().enumerate() {
            for (j, pixel) in image.pixels.iter().enumerate() {

                // rescale pixel values to be between 0 and 1
                image_matrix[[j, i]] = *pixel as f64 / 255.0;
            }

        }
        image_matrix
    }

    pub fn get_label_matrix(&self) -> Array2<f64> {
        let mut label_matrix = Array2::<f64>::zeros((10, self.images.len()));
        for (i, image) in self.images.iter().enumerate() {
            label_matrix[[image.label as usize, i]] = 1.0;
        }
        label_matrix
    }

    pub fn get_label_vector(&self) -> Vec<usize> {
        let mut label_vector = Vec::new();
        for image in self.images.iter() {
            label_vector.push(image.label as usize);
        }
        label_vector
    }
}