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
const SDR_BUF_SIZE: usize = 524288;

// Separating out the buffer sizes seems to help some.
const BUF_SIZE: usize = 262144;

// Most other sample rates fail, but this one works for my particular device.
// I will investigate exactly what's happening here and generate a list of
// possible sample rates (if there are others besides this one).
const SAMPLE_RATE: u32 = 2800000;

// Here is the sample rate of the output waveform we'll try to use.
const SAMPLE_RATE_AUDIO: f64 = 48000.0;

// The number of channels being used for audio.
const CHANNELS: i32 = 2;

fn main() {
    match run() {
        Ok(()) => (),
        Err(e) => println!("ERROR: {}", e),
    }
}

struct RTLSDR {
    rtlsdr: RTLSDRDevice,
}

unsafe impl Send for RTLSDR {}


fn run() -> Result<(), Box<Error>> {
    let sdr_index: i32 = env::args().nth(1)
        .expect("Please specify the SDR index")
        .parse::<i32>()?;
    let fm_freq_mhz: f32 = env::args().nth(2)
        .expect("Please specify a center frequency (MHz).")
        .parse::<f32>()?;
    let bandwidth: u32 = env::args().nth(3)
        .expect("Please specify the bandwidth.")
        .parse::<u32>()?;

    // Initialize the PortAudio class.
    let audio = try!(pa::PortAudio::new());
    let settings =
        try!(audio.default_output_stream_settings(CHANNELS,
                                                  SAMPLE_RATE_AUDIO,
                                                  BUF_SIZE as u32));
    let mut stream = try!(audio.open_blocking_stream(settings));
    try!(stream.start());

    let fm_freq: u32 = (fm_freq_mhz * 1e6) as u32;

    // TODO: we need to figure out the appropriate decimation here. I've seen
    // talk that we decimate based on the bandwidth but after decimation the
    // audio signal comes out much slower. Changing the decimation value
    // seems to affect the speed of the audio, so I need to establish the
    // relationship here. Just playing with the decimation value, 30 seems
    // to work well but I need to figure out _why_.
    // let dec_rate = (SAMPLE_RATE as f64 / bandwidth as f64) as usize;
    let dec_rate = 30;
    println!("dec rate is {}", dec_rate);
    let mut sdr =
        RTLSDR { rtlsdr: init_sdr(sdr_index, fm_freq, bandwidth).unwrap() };
    let mut prev = Complex::new(0.0, 0.0);


    loop {
        // Read the samples and decimate down to match the sample rate of
        // the audio that'll be going out.
        let bytes = sdr.rtlsdr.read_sync(SDR_BUF_SIZE).unwrap();
        let mut iq_vec = decimate(read_samples(bytes).unwrap().as_slice(),
                                  dec_rate);
        iq_vec = filter(iq_vec, moving_average(10));

        // After decimation, demodulate the signal and send out of the
        // thread to the receiver.
        let res = demod_fm(iq_vec, prev);
        let mut demod_iq = res.0; // filter_queue(res.0, moving_average(1));
        prev = res.1;

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
        let write_frames = std::cmp::min(buffer_frames, out_frames);
        if write_frames == 0 {
            println!("write frames is 0, are we not generating fast enough?");
        }
        let n_write_samples = write_frames as usize * CHANNELS as usize;

        stream.write(write_frames,
                   |output| for ix in 0..n_write_samples as usize {
                       output[ix] = 0.005 * demod_iq.pop_front().unwrap();
                   })?;
    }
}

fn init_sdr(sdr_index: i32,
            fm_freq: u32,
            bandwidth: u32)
            -> Result<RTLSDRDevice, RTLSDRError> {
    let mut sdr = try!(rtlsdr::open(sdr_index));
    try!(sdr.set_center_freq(fm_freq));
    try!(sdr.set_sample_rate(SAMPLE_RATE));
    try!(sdr.set_tuner_bandwidth(bandwidth));
    try!(sdr.set_agc_mode(false));
    try!(sdr.reset_buffer());
    Ok(sdr)
}

fn decimate(signal: &[Complex<f32>], dec_rate: usize) -> Vec<Complex<f32>> {
    let mut ix = 0;
    let new_size = (signal.len() / dec_rate + 1) as usize;
    let mut signal_dec = Vec::<Complex<f32>>::with_capacity(new_size);
    while ix < signal.len() {
        signal_dec.push(signal[ix]);
        ix += dec_rate;
    }
    signal_dec
}

fn read_samples(bytes: Vec<u8>) -> Option<Vec<Complex<f32>>> {
    // First, check that we've been given an even number of bytes. Not sure
    // what to do if we don't get I and Q.
    let bytes_len = bytes.len();
    if bytes_len % 2 != 0 {
        println!("Received an odd number of samples.");
        return None;

    }
    let mut iq_vec: Vec<Complex<f32>> = Vec::with_capacity(bytes_len / 2);

    // Write the values to the complex value and normalize from [0, 255] to
    // [-1, 1]. If we don't normalize, it seems we don't get anything.
    for iq in bytes.chunks(2) {
        let iq_cmplx = Complex::new((iq[0] as f32 - 127.0) / 127.0,
                                    (iq[1] as f32 - 127.0) / 127.0);
        iq_vec.push(iq_cmplx);
    }
    Some(iq_vec)
}

fn demod_fm(iq: Vec<Complex<f32>>,
            prev: Complex<f32>)
            -> (VecDeque<f32>, Complex<f32>) {

    let mut p = prev.clone();
    let mut demod_queue: VecDeque<f32> = VecDeque::with_capacity(iq.len());
    let gain = SAMPLE_RATE as f32 / (2.0 * PI * 75e3 / 8.0);

    for samp in iq.iter() {
        let conj = p.conj() * samp;
        let fm_val = conj.im.atan2(conj.re);
        demod_queue.push_back(fm_val * gain);
        p = *samp;
    }
    (demod_queue, p)
}

fn filter(samples: Vec<Complex<f32>>, taps: Vec<f32>) -> Vec<Complex<f32>> {
    let mut filt_samps: Vec<Complex<f32>> = Vec::new();
    for window in samples.as_slice().windows(taps.len()) {
        let iter = window.iter().zip(taps.iter());
        let filt_samp = iter.map(|(x, y)| x * y)
            .fold(Complex::new(0.0, 0.0), |acc, x| acc + x);
        filt_samps.push(filt_samp);
    }
    filt_samps
}

fn filter_queue(samples: VecDeque<f32>, taps: Vec<f32>) -> VecDeque<f32> {
    let mut filt_samps: VecDeque<f32> = VecDeque::new();
    for window in samples.as_slices().0.windows(taps.len()) {
        let iter = window.iter().zip(taps.iter());
        let filt_samp = iter.map(|(x, y)| x * y).sum();
        filt_samps.push_back(filt_samp);
    }
    filt_samps

}

fn moving_average(ntaps: usize) -> Vec<f32> {
    vec![1.0 / ntaps as f32; ntaps]
}
