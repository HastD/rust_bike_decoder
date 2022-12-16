use crate::{
    error::RuntimeError,
    keys::{Key, KeyFilter, WeakType},
    ncw::NearCodewordClass,
    random::Seed,
    vectors::InvalidSupport,
};
use std::{
    cmp,
    path::{Path, PathBuf},
};
use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(short='N',long,help="Number of trials (required)")]
    number: f64, // parsed as scientific notation to usize
    #[arg(short, long, default_value_t=0, value_parser=clap::value_parser!(i8).range(-1..=3),
        help="Weak key filter (-1: non-weak only; 0: no filter; 1-3: type 1-3 only)")]
    weak_keys: i8,
    #[arg(short='T',long,default_value_t=3,requires="weak_keys",help="Weak key threshold")]
    weak_key_threshold: usize,
    #[arg(long, help="Always use the specified key (in JSON format)")]
    fixed_key: Option<String>,
    #[arg(short='S',long,help="Use error vectors from near-codeword set A_{t,l}(S)")]
    ncw: Option<NearCodewordClass>,
    #[arg(short='l',long,help="Overlap parameter l in A_{t,l}(S)",requires="ncw")]
    ncw_overlap: Option<usize>,
    #[arg(short,long,help="Output file [default: stdout]")]
    output: Option<String>,
    #[arg(long, help="If output file already exists, overwrite without creating backup")]
    overwrite: bool,
    #[arg(short,long,default_value_t=10000.0,help="Max number of decoding failures recorded")]
    recordmax: f64, // parsed as scientific notation to usize
    #[arg(short,long,help="Save to disk frequency [default: only at end]")]
    savefreq: Option<f64>, // parsed as scientific notation to usize
    #[arg(long, conflicts_with="threads", help="Specify PRNG seed as 256-bit hex string [default: random]")]
    seed: Option<String>,
    #[arg(long, help="Number of threads [default: one per core]")]
    threads: Option<usize>,
    #[arg(short, long, action = clap::ArgAction::Count,
        help="Print statistics and/or decoding failures [repeat for more verbose, max 3]")]
    verbose: u8,
}

#[derive(Clone, Debug)]
pub struct Settings {
    number_of_trials: usize,
    trial_settings: TrialSettings,
    save_frequency: usize,
    record_max: usize,
    verbose: u8,
    seed: Option<Seed>,
    thread_count: usize,
    output_file: Option<PathBuf>,
    overwrite: bool,
}

impl Settings {
    const MIN_SAVE_FREQUENCY: usize = 10000;
    const MAX_THREAD_COUNT: usize = 1024;

    #[inline]
    pub fn number_of_trials(&self) -> usize {
        self.number_of_trials
    }

    #[inline]
    pub fn set_number_of_trials(&mut self, count: usize) {
        self.number_of_trials = count;
    }

    #[inline]
    pub fn trial_settings(&self) -> &TrialSettings {
        &self.trial_settings
    }

    #[inline]
    pub fn key_filter(&self) -> KeyFilter {
        self.trial_settings.key_filter()
    }

    #[inline]
    pub fn fixed_key(&self) -> Option<&Key> {
        self.trial_settings.fixed_key()
    }

    #[inline]
    pub fn ncw_class(&self) -> Option<NearCodewordClass> {
        self.trial_settings.ncw_class()
    }

    #[inline]
    pub fn ncw_overlap(&self) -> Option<usize> {
        self.trial_settings.ncw_overlap()
    }

    #[inline]
    pub fn save_frequency(&self) -> usize {
        self.save_frequency
    }

    #[inline]
    pub fn record_max(&self) -> usize {
        self.record_max
    }

    #[inline]
    pub fn verbose(&self) -> u8 {
        self.verbose
    }

    #[inline]
    pub fn seed(&self) -> Option<Seed> {
        self.seed
    }

    #[inline]
    pub fn thread_count(&self) -> usize {
        self.thread_count
    }

    #[inline]
    pub fn output_file(&self) -> Option<&Path> {
        self.output_file.as_deref()
    }

    #[inline]
    pub fn overwrite(&self) -> bool {
        self.overwrite
    }

    pub fn validate(&self) -> Result<(), RuntimeError> {
        if self.save_frequency < Self::MIN_SAVE_FREQUENCY {
            return Err(RuntimeError::RangeError(
                format!("save_frequency must be >= {}", Self::MIN_SAVE_FREQUENCY)));
        } else if self.thread_count > Self::MAX_THREAD_COUNT {
            return Err(RuntimeError::RangeError(
                format!("thread_count must be <= {}", Self::MAX_THREAD_COUNT)));
        } else if self.seed.is_some() && self.thread_count > 1 {
            return Err(RuntimeError::DependencyError(
                "seed can only be specified in single-threaded mode".to_string()));
        }
        Ok(())
    }

    pub fn from_args(args: Args) -> Result<Self, RuntimeError> {
        let settings = Self {
            number_of_trials: args.number as usize,
            trial_settings: TrialSettings::new(
                match args.weak_keys {
                    0 => KeyFilter::Any,
                    -1 => KeyFilter::NonWeak(args.weak_key_threshold),
                    1 => KeyFilter::Weak(WeakType::Type1, args.weak_key_threshold),
                    2 => KeyFilter::Weak(WeakType::Type2, args.weak_key_threshold),
                    3 => KeyFilter::Weak(WeakType::Type3, args.weak_key_threshold),
                    _ => {
                        return Err(RuntimeError::RangeError(
                            "weak_key_filter must be in {-1, 0, 1, 2, 3}".to_string()));
                    }
                },
                args.fixed_key.as_deref().map(serde_json::from_str).transpose()?.map(Key::sorted),
                args.ncw,
                args.ncw_overlap
            )?,
            save_frequency: cmp::max(Self::MIN_SAVE_FREQUENCY, args.savefreq.unwrap_or(args.number) as usize),
            record_max: args.recordmax as usize,
            verbose: args.verbose,
            seed: args.seed.map(Seed::try_from).transpose()?,
            thread_count: args.threads.map_or_else(|| num_cpus::get_physical(),
                |threads| cmp::min(cmp::max(threads, 1), Self::MAX_THREAD_COUNT)),
            output_file: args.output.map(PathBuf::from),
            overwrite: args.overwrite,
        };
        settings.validate()?;
        Ok(settings)
    }
}

#[derive(Clone, Debug, Default)]
pub struct TrialSettings {
    key_filter: KeyFilter,
    fixed_key: Option<Key>,
    ncw_class: Option<NearCodewordClass>,
    ncw_overlap: Option<usize>,
}

impl TrialSettings {
    pub fn new(
        key_filter: KeyFilter,
        fixed_key: Option<Key>,
        ncw_class: Option<NearCodewordClass>,
        ncw_overlap: Option<usize>    
    ) -> Result<Self, RuntimeError> {
        if let Some(key) = fixed_key.as_ref() {
            key.validate()?;
            if !key.matches_filter(key_filter) {
                return Err(RuntimeError::DataError(InvalidSupport(
                    "fixed_key does not match key filter".to_string())));
            }
        }
        if let Some(l) = ncw_overlap {
            let sample_class = ncw_class.ok_or_else(|| RuntimeError::DependencyError(
                "ncw_overlap requires ncw_class to be set".to_string()))?;
            let l_max = sample_class.max_l();
            if l > l_max {
                return Err(RuntimeError::RangeError(
                    format!("l must be in range 0..{} in A_{{t,l}}({})", l_max, sample_class)));
            }
        }
        Ok(Self { key_filter, fixed_key, ncw_class, ncw_overlap })
    }

    #[inline]
    pub fn key_filter(&self) -> KeyFilter {
        self.key_filter
    }

    #[inline]
    pub fn fixed_key(&self) -> Option<&Key> {
        self.fixed_key.as_ref()
    }

    #[inline]
    pub fn ncw_class(&self) -> Option<NearCodewordClass> {
        self.ncw_class
    }

    #[inline]
    pub fn ncw_overlap(&self) -> Option<usize> {
        self.ncw_overlap
    }
}
