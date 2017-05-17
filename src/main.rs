extern crate rtlsdr;
extern crate num;

use rtlsdr::{RTLSDRDevice, RTLSDRError};
use num::complex::Complex;
use std::env;

fn main() {
    let sdr_index: i32 = env::args().nth(1)
        .expect("Please specify the SDR index")
        .parse::<i32>()
        .unwrap();
    let fm_freq_mhz: f32 = env::args().nth(2)
        .expect("Please specify a center frequency (MHz).")
        .parse::<f32>()
        .unwrap();

    let fm_freq: u32 = (fm_freq_mhz * 1e6) as u32;
    let mut sdr = init_sdr(sdr_index, fm_freq);
}

fn init_sdr(sdr_index: i32, fm_freq: u32) -> Result<RTLSDRDevice, RTLSDRError> {
    let mut sdr = try!(rtlsdr::open(sdr_index));
    try!(sdr.set_center_freq(fm_freq));
    Ok(sdr)
}

fn read_samples(bytes: Vec<u8>) -> Option<Complex<f32>> {
    // The docs tell us that RTL-SDR sends us data in bytes alternating between
    // I and Q. I'm not normalizing here, but I might have to in the future.
    
    // If we read an odd number of bytes, I don't think we can do anything.
    // (Make sure that this can actually happen)./
    let iter = bytes.chunks(2);
    if iter.count() % 2 != 0 {
        return None
    }

    // Right now I'm just putting None here so we'll compile. 
    None
}
