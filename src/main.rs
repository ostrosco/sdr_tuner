extern crate num;
extern crate portaudio;
extern crate rtlsdr;

use num::complex::Complex;
use num::{Num, Zero};
use pa::error::Error as PAError;
use portaudio as pa;
use rtlsdr::{RTLSDRDevice, RTLSDRError};
use std::cmp::min;
use std::env;
use std::error::Error;
use std::f32::consts::PI;
use std::sync::mpsc::channel;
use std::thread;

// Make sure that the buffer size is radix-2, otherwise the read_sync function
// will fail with an error code of -8. I want the buffer sizes between the SDR
// and the audio sizes to match, but since we get two samples per read the
// buffer size needs to be double.
const SDR_BUF_SIZE: usize = 262_144;

// The buffer size for the audio sink.
const AUDIO_BUF_SIZE: usize = 524_288;

// Most other sample rates fail, but this one works for my particular device.
// I will investigate exactly what's happening here and generate a list of
// possible sample rates (if there are others besides this one).
const SDR_SAMPLE_RATE: u32 = 2_400_000;

// Here is the sample rate of the output waveform we'll try to use.
const AUDIO_SAMPLE_RATE: f32 = 48000.0;

// The number of channels being used for audio.
const CHANNELS: i32 = 2;

// We set the gain manually right now since it seems to be a better option
// than using the AGC (the SDR gets blazing hot with the AGC on).
const SDR_GAIN: i32 = 496;

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
    let sdr_index: i32 = env::args()
        .nth(1)
        .expect("Please specify the SDR index.")
        .parse::<i32>()?;
    let fm_freq_mhz: f32 = env::args()
        .nth(2)
        .expect("Please specify a center frequency in MHz.")
        .parse::<f32>()?;

    // Set up the threading channels for communication between our two threads.
    let (dongle_tx, dongle_rx) = channel();
    let (demod_tx, demod_rx) = channel();

    // This thread only collects data off of the SDR as quickly as possible and
    // sends it out.
    thread::spawn(move || {
        // Initialize the SDR based on the user's input parameters.
        let fm_freq: u32 = (fm_freq_mhz * 1e6) as u32;
        let mut sdr = RTLSDR {
            rtlsdr: init_sdr(sdr_index, fm_freq).unwrap(),
        };

        loop {
            let bytes = sdr.rtlsdr.read_sync(SDR_BUF_SIZE).unwrap();
            let iq_vec = samps_to_cmplx(bytes.as_slice()).unwrap();
            dongle_tx.send(iq_vec).unwrap();
        }
    });

    // This thread does all of the signal processing required to demodulate
    // the signal and get it to a nice audio stream.
    thread::spawn(move || {
        // We decimate down to 100k samples. This is close to the optimal value
        // of 96k samples, but provides just enough buffer to avoid overruns and
        // improves the audio quality substantially.
        let dec_rate_filt = 6;
        let dec_rate_audio = 4;

        let taps_filt =
            windowed_sinc(SDR_SAMPLE_RATE as f32, 16, &hamming);
        let taps_audio = windowed_sinc(
            SDR_SAMPLE_RATE as f32 / dec_rate_filt as f32,
            16,
            &hamming,
        );

        let mut prev = Complex::zero();
        loop {
            // First, low_pass and decimate the raw IQ signal.
            let mut iq_vec = dongle_rx.recv().unwrap();
            iq_vec = filter(&iq_vec, &taps_filt);
            iq_vec = decimate(&iq_vec, dec_rate_filt);

            // Next, demodulate the signal and filter the signal again. The
            // second filter seems to help with the noise after demodulation.
            let res = demod_fm(&iq_vec, prev);
            let mut demod_iq = filter_real(&res.0, &taps_audio);
            demod_iq = decimate(&demod_iq, dec_rate_audio);
            prev = res.1;
            demod_tx.send(demod_iq).unwrap();
        }
    });

    // Initialize the PortAudio class. We use a blocking stream since we're
    // doing processing on the signal and can't keep up with non-blocking
    // timing requirements.
    let audio = try!(pa::PortAudio::new());
    let settings = try!(audio.default_output_stream_settings(
        CHANNELS,
        AUDIO_SAMPLE_RATE as f64,
        AUDIO_BUF_SIZE as u32
    ));
    let mut stream = try!(audio.open_blocking_stream(settings));
    try!(stream.start());

    loop {
        let demod_iq = demod_rx.recv().unwrap();
        let buffer_frames = (demod_iq.len() / CHANNELS as usize) as u32;
        let mut out_frames = 0;

        while out_frames < buffer_frames {
            out_frames = match stream.write_available() {
                Ok(available) => match available {
                    pa::StreamAvailable::Frames(frames) => frames as u32,
                    pa::StreamAvailable::InputOverflowed => {
                        return Err(Box::new(PAError::InputOverflowed))
                    }
                    pa::StreamAvailable::OutputUnderflowed => {
                        return Err(Box::new(PAError::OutputUnderflowed))
                    }
                },
                Err(e) => return Err(Box::new(e)),
            };
        }

        let n_write_samples = buffer_frames as usize * CHANNELS as usize;

        stream.write(buffer_frames, |output| {
            for ix in 0..n_write_samples {
                output[ix] = demod_iq[ix];
            }
        })?;
    }
}

fn init_sdr(sdr_index: i32, fm_freq: u32) -> Result<RTLSDRDevice, RTLSDRError> {
    let mut sdr = try!(rtlsdr::open(sdr_index));
    try!(sdr.set_center_freq(fm_freq));
    try!(sdr.set_sample_rate(SDR_SAMPLE_RATE));
    try!(sdr.set_tuner_gain(SDR_GAIN));
    try!(sdr.set_agc_mode(false));
    try!(sdr.reset_buffer());
    Ok(sdr)
}

/// Decimates a signal by an integer factor.
///
/// # Arguments
///
/// * `signal` - The signal to decimate
/// * `dec_rate` - The decimation factor
///
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

/// Converts samples from the SDR into complex IQ pairs.
///
/// # Arguments
///
/// * `bytes` - The unsigned bytes from the SDR.
///
fn samps_to_cmplx(bytes: &[u8]) -> Option<Vec<Complex<f32>>> {
    // First, check that we've been given an even number of bytes. Not sure
    // what to do if we don't get I and Q.
    let bytes_len = bytes.len();
    if bytes_len % 2 != 0 {
        println!("Received an odd number of samples.");
        return None;
    }

    // Write the values to the complex value and normalize from [0, 255] to
    // [-1, 1].
    let mut iq_vec: Vec<Complex<f32>> = Vec::with_capacity(bytes_len / 2);
    for iq in bytes.chunks(2) {
        let iq_cmplx = Complex::new(
            (iq[0] as f32 - 127.0) / 127.0,
            (iq[1] as f32 - 127.0) / 127.0,
        );
        iq_vec.push(iq_cmplx);
    }
    Some(iq_vec)
}

/// Performs FM demodulation on a complex signal.
///
/// # Arguments
///
/// * `iq` - The IQ samples to demodulate.
/// * `prev` - The previous sample from the previous demodulation.
/// * `deviation` - The maximum frequency deviation in Hz.
///
fn demod_fm(
    iq: &[Complex<f32>],
    prev: Complex<f32>,
) -> (Vec<f32>, Complex<f32>) {
    let mut p = prev;
    let mut demod_queue: Vec<f32> = Vec::with_capacity(iq.len());

    // Magic gain number. Would be nice to get something better.
    let gain = SDR_SAMPLE_RATE as f32 / (2.0 * PI * 75.0);

    for samp in iq {
        let conj = p.conj() * samp;
        demod_queue.push(conj.arg() * gain);
        p = *samp;
    }
    (demod_queue, p)
}

/// Filters a Complex vector and returns a new vector.
///
/// # Arguments
///
/// * `samples` - The complex samples to filter
/// * `taps` - The taps of the filter
///
fn filter<T: Copy + Num + Zero>(
    samples: &[Complex<T>],
    taps: &[T],
) -> Vec<Complex<T>> {
    let mut filt_samps: Vec<Complex<T>> = Vec::with_capacity(samples.len());

    for samp in 0..samples.len() {
        let mut filt_val = Complex::zero();
        for filt in 0..min(samp, taps.len()) {
            filt_val = filt_val + samples[samp - filt] * taps[filt];
        }
        filt_samps.push(filt_val);
    }

    filt_samps
}

/// Filters a real vector and returns a new vector.
///
/// # Arguments
///
/// * `samples` - The real samples to filter
/// * `taps` - The taps of the filter
///
fn filter_real<T: Copy + Num + Zero>(samples: &[T], taps: &[T]) -> Vec<T> {
    let mut filt_samps: Vec<T> = Vec::with_capacity(samples.len());

    for samp in 0..samples.len() {
        let mut filt_val = T::zero();
        for filt in 0..min(samp, taps.len()) {
            filt_val = filt_val + samples[samp - filt] * taps[filt];
        }
        filt_samps.push(filt_val);
    }

    filt_samps
}

/// Generates a vector containing the taps for a Hamming window.
///
/// # Arguments
///
/// * `ntaps` - The number of taps to generate for the filter
///
fn hamming(ntaps: usize) -> Vec<f32> {
    let m: f32 = ntaps as f32 - 1.0;
    let mut taps: Vec<f32> = Vec::with_capacity(ntaps);
    for ix in 0..ntaps {
        taps.push(0.54 - 0.46 * (2.0 * PI * ix as f32 / m).cos());
    }
    taps
}

/// Generates filter taps for a sinc filter with a windowing function.
///
/// # Arguments
///
/// * `sample_rate` - The sample frequency in Hertz
/// * `ntaps` - The number of taps to generate for the window
/// * `window` - The windowing function used to generate the taps for the window
///
fn windowed_sinc(
    sample_rate: f32,
    ntaps: usize,
    window: &Fn(usize) -> Vec<f32>,
) -> Vec<f32> {
    let wc: f32 = 2.0 * PI / sample_rate;

    let win_taps: Vec<f32> = window(ntaps);
    let mut taps: Vec<f32> = Vec::with_capacity(ntaps);

    let ntaps_i: i32 = ntaps as i32;
    let m: i32 = (ntaps_i - 1) / 2;

    for ix in -m..m + 1 {
        let ix_f = ix as f32;

        // The windowed sinc function is undefined at zero, so set it to
        // something sensible here.
        let mut hi = if ix == 0 {
            wc
        } else {
            (ix_f * wc).sin() / (ix_f)
        };
        hi *= win_taps[(ix + m) as usize];
        taps.push(hi);
    }
    taps
}
