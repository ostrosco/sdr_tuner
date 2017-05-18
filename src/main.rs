extern crate rtlsdr;
extern crate num;

use rtlsdr::{RTLSDRDevice, RTLSDRError};
use num::complex::Complex;
use std::env;

// Make sure that the buffer size is radix-2, otherwise the read_sync function
// will fail with an error code of -8.
const BUF_SIZE: usize = 131072;

// Most other sample rates fail, but this one works for my particular device.
// I will investigate exactly what's happening here and generate a list of
// possible sample rates (if there are others besides this one).
const SAMPLE_RATE: u32 = 2048000;

fn main() {
    let sdr_index: i32 = env::args()
        .nth(1)
        .expect("Please specify the SDR index")
        .parse::<i32>()
        .unwrap();
    let fm_freq_mhz: f32 = env::args()
        .nth(2)
        .expect("Please specify a center frequency (MHz).")
        .parse::<f32>()
        .unwrap();

    let fm_freq: u32 = (fm_freq_mhz * 1e6) as u32;
    let mut sdr = match init_sdr(sdr_index, fm_freq) {
        Ok(s) => s,
        Err(e) => return println!("Couldn't initialize SDR, {}", e),
    };

    loop {
        let bytes = match sdr.read_sync(BUF_SIZE) {
            Ok(b) => b,
            Err(e) => return println!("Couldn't read bytes from SDR, {}", e),
        };
        let iq_vec = read_samples(bytes);
        match iq_vec {
            // TODO: Here is where we would start demodulating. For now, just
            // leave it be.
            Some(iq) => println!("We got some bytes: {:?}", iq.get(0)),
            None => println!("No bytes. :("),
        }
    }
}

fn init_sdr(sdr_index: i32, fm_freq: u32) -> Result<RTLSDRDevice, RTLSDRError> {
    let mut sdr = try!(rtlsdr::open(sdr_index));
    try!(sdr.set_center_freq(fm_freq));
    try!(sdr.set_sample_rate(SAMPLE_RATE));
    try!(sdr.reset_buffer());
    Ok(sdr)
}

fn read_samples(bytes: Vec<u8>) -> Option<Vec<Complex<f32>>> {
    // The docs tell us that RTL-SDR sends us data in bytes alternating between
    // I and Q. I'm not normalizing here, but I might have to in the future.

    // First, check that we've been given an even number of bytes. Not sure
    // what to do if we don't get I and Q.
    let bytes_len = bytes.len();
    if bytes_len % 2 != 0 {
        return None;

    }
    let mut iter = bytes.chunks(2);
    let mut iq_vec: Vec<Complex<f32>> = Vec::with_capacity(bytes_len / 2);

    // Write the values to the complex value and normalize from [0, 255] to
    // [-1, 1].
    for iq in iter.next() {
        iq_vec.push(Complex::new((iq[0] as f32 - 127.5) / 127.5,
                                 (iq[1] as f32 - 127.5) / 127.5));
    }

    Some(iq_vec)
}
