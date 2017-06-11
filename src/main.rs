extern crate num;
extern crate portaudio;
extern crate rtlsdr;
extern crate dft;
extern crate gnuplot; 

use portaudio as pa;
use rtlsdr::{RTLSDRDevice, RTLSDRError};
use num::Zero;
use num::complex::Complex;
use std::env;
use std::f32::consts::PI;
use std::error::Error;

use gnuplot::*;
use dft::{Operation, Plan};

// Make sure that the buffer size is radix-2, otherwise the read_sync function
// will fail with an error code of -8.
const SDR_BUF_SIZE: usize = 262144;

// Separating out the buffer sizes seems to help some.
const BUF_SIZE: usize = 262144;

// Most other sample rates fail, but this one works for my particular device.
// I will investigate exactly what's happening here and generate a list of
// possible sample rates (if there are others besides this one).
const SAMPLE_RATE: u32 = 2400000;

// Here is the sample rate of the output waveform we'll try to use.
const SAMPLE_RATE_AUDIO: f32 = 44100.0;

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
                                                  SAMPLE_RATE_AUDIO as f64,
                                                  BUF_SIZE as u32));
    let mut stream = try!(audio.open_blocking_stream(settings));
    try!(stream.start());

    let fm_freq: u32 = (fm_freq_mhz * 1e6) as u32;

    // We were decimating way too much. We need to decimate in two stages: one
    // to remove nearby stations and another to match the sound card. For
    // wideband signals, 8 and 3 seems to work well and also hits that sweet
    // spot of no buffer underflows! 
    //
    // TODO: need to still calculate this on the fly. In addition, we need to
    // find apporpriate decimation values for narrowband. I'm still confident
    // that the decimation value depends on the bandwitdh, but I need to
    // figure out that appropriate formula.
    //
    let dec_rate_filt = 8;
    let dec_rate_audio = 3;
    let mut sdr =
        RTLSDR { rtlsdr: init_sdr(sdr_index, fm_freq, bandwidth).unwrap() };

    // This holds the previous value for the moving average filter so we can
    // keep track in between iterations of reading from the SDR.
    let mut prev = Complex::new(0.0, 0.0);
    let taps = low_pass(bandwidth as f32, SAMPLE_RATE as f32, 8);

    loop {
        // Read the samples and decimate down to match the sample rate of
        // the audio that'll be going out.
        let bytes = sdr.rtlsdr.read_sync(SDR_BUF_SIZE).unwrap();
        let mut iq_vec = read_samples(bytes).unwrap();
        // plot_spectrum(&iq_vec, &mut figure1);
        iq_vec = filter(iq_vec, taps.clone());
        iq_vec = decimate(iq_vec.as_slice(), dec_rate_filt);

        // After decimation, demodulate the signal and send out of the
        // thread to the receiver.
        let res = demod_fm(iq_vec, prev);
        let mut demod_iq = filter_real(res.0, taps.clone());
        demod_iq = decimate(demod_iq.as_slice(), dec_rate_audio);
        prev = res.1;

        // Get the write stream and write our samples to the stream.
        let out_frames = match stream.write_available() {
            Ok(available) => {
                match available {
                    pa::StreamAvailable::Frames(frames) => frames as u32,
                    pa::StreamAvailable::OutputUnderflowed => return Err(Box::new(pa::error::Error::OutputUnderflowed)),
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
                       output[ix] = 0.005 * demod_iq[ix];
                   })?;
    }
}

fn plot_spectrum<'a>(signal: &Vec<Complex<f32>>, figure: &'a mut Figure) -> &'a mut Figure {
    let mut sig = signal.clone();
    let radix_2: u32 = (sig.len() as f32).log2().ceil() as u32;
    let new_len = 2u32.pow(radix_2) as usize;
    println!("signal len: {}, new len: {}", sig.len(), new_len);
    sig.resize(new_len, Complex::zero());
    let plan = Plan::new(Operation::Forward, new_len);
    dft::transform(&mut sig, &plan);

    // Now generate a plot.
    figure.clear_axes();
    figure.axes2d()
        .lines(0..new_len, sig.iter().map(|x| x.norm()), &[]);
    figure.show();
    figure
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

fn decimate<T: Copy>(signal: &[T], dec_rate: usize) -> Vec<T> {
    let mut ix = 0;
    let new_size = (signal.len() / dec_rate + 1) as usize;
    let mut signal_dec = Vec::<T>::with_capacity(new_size);
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
            -> (Vec<f32>, Complex<f32>) {

    let mut p = prev.clone();
    let mut demod_queue: Vec<f32> = Vec::with_capacity(iq.len());
    let gain = SAMPLE_RATE as f32 / (2.0 * PI * 75e3 / 8.0);

    for samp in iq.iter() {
        let conj = p.conj() * samp;
        demod_queue.push(conj.arg() * gain);
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

fn filter_real(samples: Vec<f32>, taps: Vec<f32>) -> Vec<f32> {
    let mut filt_samps: Vec<f32> = Vec::new();
    for window in samples.as_slice().windows(taps.len()) {
        let iter = window.iter().zip(taps.iter());
        let filt_samp = iter.map(|(x, y)| x * y)
            .fold(0.0, |acc, x| acc + x);
        filt_samps.push(filt_samp);
    }
    filt_samps
}

fn hamming(ntaps: usize) -> Vec<f32> {
    let m: f32 = ntaps as f32 - 1.0;
    let mut taps: Vec<f32> = Vec::with_capacity(ntaps);
    for ix in 0..ntaps {
        taps.push(0.54 - 0.46 * (2.0 * PI * ix as f32 / m).cos());
    }
    taps
}

fn low_pass(cutoff_freq: f32, sample_rate: f32, ntaps: usize) -> Vec<f32> {
    let wc: f32 = 2.0 * PI * cutoff_freq / sample_rate;
    let mut taps: Vec<f32> = Vec::with_capacity(ntaps);
    let win_taps: Vec<f32> = hamming(ntaps);
    let ntaps_i: i32 = ntaps as i32;
    let m: i32 = (ntaps_i - 1) / 2;

    for ix in -m..m + 1 {
        let ix_f: f32 = ix as f32;
        if ix == 0 {
            let hi = wc / PI * win_taps[(ix + m) as usize];
            taps.push(hi);
        } else {
            let hi = (ix_f * wc).sin() / (ix_f * PI) *
                     win_taps[(ix + m) as usize];
            taps.push(hi);
        }
    }
    taps
}

#[inline]
fn moving_average(ntaps: usize) -> Vec<f32> {
    vec![1.0 / ntaps as f32; ntaps]
}
