extern crate num;
extern crate portaudio;
extern crate rtlsdr;

use portaudio as pa;
use rtlsdr::{RTLSDRDevice, RTLSDRError};
use num::complex::Complex;
use std::env;
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::error::Error;

// Make sure that the buffer size is radix-2, otherwise the read_sync function
// will fail with an error code of -8.
const BUF_SIZE: usize = 131072;

// Most other sample rates fail, but this one works for my particular device.
// I will investigate exactly what's happening here and generate a list of
// possible sample rates (if there are others besides this one).
const SAMPLE_RATE: u32 = 1800000;

// Here is the sample rate of the output waveform we'll try to use.
const SAMPLE_RATE_FREQ: f64 = 48000.0;

const CHANNELS: i32 = 2;

const DEVIATION: i32 = 2500;

const BANDWIDTH: u32 = 25000;

fn main() {
    match run() {
        Ok(()) => (),
        Err(e) => println!("ERROR: {}", e),
    }
}

fn run() -> Result<(), Box<Error>> {
    let sdr_index: i32 = env::args().nth(1)
        .expect("Please specify the SDR index")
        .parse::<i32>()?;
    let fm_freq_mhz: f32 = env::args().nth(2)
        .expect("Please specify a center frequency (MHz).")
        .parse::<f32>()?;

    // Initialize the PortAudio class.
    let pa = pa::PortAudio::new()?;
    let settings = pa.default_output_stream_settings(CHANNELS,
                                                     SAMPLE_RATE_FREQ,
                                                     BUF_SIZE as u32)
        .unwrap();
    let mut stream = pa.open_blocking_stream(settings)?;
    try!(stream.start());

    let fm_freq: u32 = (fm_freq_mhz * 1e6) as u32;
    let mut sdr = init_sdr(sdr_index, fm_freq).unwrap();
    let gain = (2.0 * PI * DEVIATION as f32 / SAMPLE_RATE_FREQ as f32).recip();

    loop {
        // Read the samples, convert to copmlex values, and demodulate them.
        let bytes = sdr.read_sync(BUF_SIZE).unwrap();
        let iq_vec = read_samples(bytes).unwrap();
        let mut demod_iq = demod_fm(iq_vec, gain);

        // Get the write stream and write our samples to the stream.
        let out_frames = match stream.write_available() {
            Ok(available) => {
                match available {
                    pa::StreamAvailable::Frames(frames) => frames as u32,
                    _ => return Err(Box::new(pa::error::Error::NullCallback)),
                }
            }
            Err(e) => return Err(Box::new(e)),
        };

        let buffer_frames = (demod_iq.len() / CHANNELS as usize) as u32;
        let write_frames = if buffer_frames >= out_frames {
            out_frames
        } else {
            buffer_frames
        };
        let n_write_samples = write_frames as usize * CHANNELS as usize;

        stream.write(write_frames,
                   |output| for ix in 0..n_write_samples as usize {
                       output[ix] = demod_iq.pop_front().unwrap();
                   })?;
    }
}

fn init_sdr(sdr_index: i32, fm_freq: u32) -> Result<RTLSDRDevice, RTLSDRError> {
    let mut sdr = try!(rtlsdr::open(sdr_index));
    try!(sdr.set_center_freq(fm_freq));
    try!(sdr.set_sample_rate(SAMPLE_RATE));
    try!(sdr.set_tuner_bandwidth(BANDWIDTH));
    try!(sdr.set_agc_mode(true));
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
        println!("Received an odd number of samples.");
        return None;

    }
    let mut iq_vec: Vec<Complex<f32>> = Vec::with_capacity(bytes_len / 2);

    // Write the values to the complex value and normalize from [0, 255] to
    // [-1, 1].
    for iq in bytes.chunks(2) {
        iq_vec.push(Complex::new((iq[0] as f32 - 127.5) / 127.5,
                                 (iq[1] as f32 - 127.5) / 127.5));
    }

    Some(iq_vec)
}

fn demod_fm(iq: Vec<Complex<f32>>, gain: f32) -> VecDeque<f32> {
    let mut demod_queue: VecDeque<f32> = VecDeque::with_capacity(iq.len());
    let mut prev = iq[0];
    for ix in 0..iq.len() {
        let c: Complex<f32> = prev.conj() * iq[ix];
        demod_queue.push_back(c.im.atan2(c.re) * gain);
        prev = iq[ix];
    }
    demod_queue
}
