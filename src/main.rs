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

// Threading libraries.
use std::thread;
use std::sync::mpsc::channel;

// Make sure that the buffer size is radix-2, otherwise the read_sync function
// will fail with an error code of -8.
const BUF_SIZE: usize = 131072;

// Most other sample rates fail, but this one works for my particular device.
// I will investigate exactly what's happening here and generate a list of
// possible sample rates (if there are others besides this one).
const SAMPLE_RATE: u32 = 1140000;

// Here is the sample rate of the output waveform we'll try to use.
const SAMPLE_RATE_AUDIO: f64 = 44100.0;

const DEVIATION: u32 = 2500;

const BANDWIDTH: u32 = 200000;

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

    // Initialize the PortAudio class.
    let audio = try!(pa::PortAudio::new());
    let settings =
        try!(audio.default_output_stream_settings(CHANNELS,
                                                  SAMPLE_RATE_AUDIO,
                                                  BUF_SIZE as u32));
    let mut stream = try!(audio.open_blocking_stream(settings));
    try!(stream.start());

    let fm_freq: u32 = (fm_freq_mhz * 1e6) as u32;
    let mut sdr = RTLSDR {
        rtlsdr: init_sdr(sdr_index, fm_freq).unwrap(),
    };

    // TODO: does this gain value make sense based on what I'm doing?
    let gain = (2.0 * PI * DEVIATION as f32 / SAMPLE_RATE_AUDIO as f32).recip();
    let (tx, rx) = channel();

    thread::spawn(move || {
        loop {
            // Read the samples and decimate down to match the sample rate of the
            // audio that'll be going out. 
            //
            // TODO: on 88.7 I can make out some instruments, but the audio is still
            // very, very poor. Need to figure out what we need to do here. At a
            // minimum, we need a low-pass filter to filter out the high
            // frequency components that are almost certainly aliasing. In addition,
            // we are underflowing the audio buffer. Not sure if this is a 
            // performance issue (release didn't help) or if I need to re-think how
            // I send data to the sound card.
            let tx = tx.clone();
            let bytes = sdr.rtlsdr.read_sync(BUF_SIZE).unwrap();
            let dec_rate = (SAMPLE_RATE as f64/ SAMPLE_RATE_AUDIO) as usize;
            let iq_vec = decimate(read_samples(bytes).unwrap().as_slice(), dec_rate);

            // After decimation, demodulate the signal. 
            //
            // TODO: from what I've read there should be some decimation here as
            // well but with narrowband FM, we decimate the signal into 
            // nothingness. Need to figure out what we do between wideband and
            // narrowband FM.
            let mut demod_iq = demod_fm(iq_vec, gain);
            tx.send(demod_iq).unwrap();
        }
    });

    loop {
        // Get the write stream and write our samples to the stream.
        let mut demod_iq = rx.recv().unwrap();
        let out_frames = match stream.write_available() {
            Ok(available) => {
                match available {
                    pa::StreamAvailable::Frames(frames) => frames as u32,
                    _ => return Err(Box::new(pa::error::Error::NullCallback)),
                }
            }
            Err(e) => return Err(Box::new(e)),
        };

        let buffer_frames = (demod_iq.len() / CHANNELS as usize) as
                            u32;
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

fn decimate_queue<T: Copy>(signal: VecDeque<T>, dec_rate: usize) -> VecDeque<T> {
    let mut ix = 0;
    let mut signal_dec = VecDeque::<T>::new();
    while ix < signal.len() {
        signal_dec.push_back(*(signal.get(ix).unwrap()));
        ix += dec_rate;
    }
    signal_dec
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
        demod_queue.push_back(c.re.atan2(c.im) * gain);
        prev = iq[ix];
    }
    demod_queue
}
