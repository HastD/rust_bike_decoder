use crate::{
    keys::{Key, KeyFilter, WeakType},
    ncw::NearCodewordClass,
    random::Seed,
};
use std::{
    cmp,
    path::{Path, PathBuf},
};
use anyhow::{Context, Result};
use clap::Parser;
use derive_builder::Builder;
use thiserror::Error;

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
    #[arg(long, help="Run in parallel with automatically chosen number of threads")]
    parallel: bool,
    #[arg(short,long,default_value_t=10000.0,help="Max number of decoding failures recorded")]
    recordmax: f64, // parsed as scientific notation to usize
    #[arg(short,long,help="Save to disk frequency [default: only at end]")]
    savefreq: Option<f64>, // parsed as scientific notation to usize
    #[arg(long, help="Specify PRNG seed as 256-bit hex string [default: random]")]
    seed: Option<String>,
    #[arg(long, help="Set number of threads (ignores --parallel)")]
    threads: Option<usize>,
    #[arg(short, long, action = clap::ArgAction::Count,
        help="Print statistics and/or decoding failures [repeat for more verbose, max 3]")]
    verbose: u8,
}

#[derive(Builder, Clone, Debug, PartialEq, Eq)]
pub struct Settings {
    number_of_trials: usize,
    #[builder(default)] trial_settings: TrialSettings,
    #[builder(default)] save_frequency: usize,
    #[builder(default="10000")] record_max: usize,
    #[builder(default)] verbose: u8,
    #[builder(default)] seed: Option<Seed>,
    #[builder(default="1")] threads: usize,
    #[builder(default)] output_file: Option<PathBuf>,
    #[builder(default="false")] overwrite: bool,
    #[builder(default="false")] silent: bool,
}

impl Settings {
    const MIN_SAVE_FREQUENCY: usize = 10000;
    const MAX_THREAD_COUNT: usize = 1024;

    pub fn from_args(args: Args) -> Result<Self> {
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
                        return Err(SettingsError::InvalidFilter.into());
                    }
                },
                args.fixed_key.as_deref().map(serde_json::from_str).transpose()
                    .context("--fixed-key should be valid JSON representing a key")?
                    .map(Key::sorted),
                args.ncw,
                args.ncw_overlap
            )?,
            save_frequency: cmp::max(Self::MIN_SAVE_FREQUENCY, args.savefreq.unwrap_or(args.number) as usize),
            record_max: args.recordmax as usize,
            verbose: args.verbose,
            seed: args.seed.map(Seed::try_from).transpose()
                .context("--seed should be 256-bit hex string")?,
            threads: args.threads.map_or_else(
                || if args.parallel { 0 } else { 1 },
                |threads| cmp::min(cmp::max(threads, 1), Self::MAX_THREAD_COUNT)),
            output_file: args.output.map(PathBuf::from),
            overwrite: args.overwrite,
            silent: false,
        };
        Ok(settings)
    }

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
        if self.save_frequency == 0 {
            self.number_of_trials
        } else {
            self.save_frequency
        }
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
    pub fn parallel(&self) -> bool {
        self.threads != 1
    }

    #[inline]
    pub fn threads(&self) -> usize {
        self.threads
    }

    #[inline]
    pub fn output_file(&self) -> Option<&Path> {
        self.output_file.as_deref()
    }

    #[inline]
    pub fn overwrite(&self) -> bool {
        self.overwrite
    }

    #[inline]
    pub fn silent(&self) -> bool {
        self.silent
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
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
    ) -> Result<Self> {
        if let Some(key) = fixed_key.as_ref() {
            key.validate().context("--fixed-key must specify valid key support")?;
            if !key.matches_filter(key_filter) {
                return Err(SettingsError::InvalidFixedKey.into());
            }
        }
        if let Some(l) = ncw_overlap {
            let sample_class = ncw_class.ok_or(SettingsError::NcwDependency)?;
            let l_max = sample_class.max_l();
            if l > l_max {
                return Err(SettingsError::NcwRange(sample_class).into());
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

#[derive(Copy, Clone, Debug, Error)]
pub enum SettingsError {
    #[error("weak_key_filter must be in {{-1, 0, 1, 2, 3}}")]
    InvalidFilter,
    #[error("fixed_key does not match key filter")]
    InvalidFixedKey,
    #[error("ncw_overlap requires ncw_class to be set")]
    NcwDependency,
    #[error("l must be in range 0..{} in A_{{t,l}}({0})", .0.max_l())]
    NcwRange(NearCodewordClass),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_args_example() {
        let args = Args {
            number: 1.75e4,
            weak_keys: -1,
            weak_key_threshold: 4,
            fixed_key: Some(r#"{"h0":[6,25,77,145,165,212,230,232,247,261,306,341,449,466,493],
                "h1":[35,108,119,159,160,163,221,246,249,286,310,360,484,559,580]}"#.to_string()),
            ncw: Some(NearCodewordClass::C),
            ncw_overlap: Some(7),
            output: Some("test/path/to/file.json".to_string()),
            overwrite: true,
            parallel: true,
            recordmax: 123.4,
            savefreq: Some(50.0),
            seed: Some("874a5940435d8a5462d8579af9f4cad2a737880dfb13620c5257a60ffaaae6cf".to_string()),
            threads: Some(usize::MAX),
            verbose: 2,
        };
        let settings = Settings::from_args(args).unwrap();
        assert_eq!(settings.number_of_trials, 17500);
        assert_eq!(settings.trial_settings.key_filter, KeyFilter::NonWeak(4));
        assert_eq!(settings.trial_settings.fixed_key, Some(Key::from_support(
            [6,25,77,145,165,212,230,232,247,261,306,341,449,466,493],
            [35,108,119,159,160,163,221,246,249,286,310,360,484,559,580]).unwrap()));
        assert_eq!(settings.trial_settings.ncw_class, Some(NearCodewordClass::C));
        assert_eq!(settings.trial_settings.ncw_overlap, Some(7));
        assert_eq!(settings.save_frequency, Settings::MIN_SAVE_FREQUENCY);
        assert_eq!(settings.record_max, 123);
        assert_eq!(settings.verbose, 2);
        assert_eq!(settings.seed, Some(Seed::from(
            [135,74,89,64,67,93,138,84,98,216,87,154,249,244,202,210,
            167,55,136,13,251,19,98,12,82,87,166,15,250,170,230,207])));
        assert_eq!(settings.threads, Settings::MAX_THREAD_COUNT);
        assert_eq!(settings.output_file, Some(PathBuf::from("test/path/to/file.json")));
        assert_eq!(settings.overwrite, true);
        assert_eq!(settings.silent, false);
    }

    #[test]
    fn settings_builder() {
        let settings = SettingsBuilder::default()
            .number_of_trials(12345).silent(true)
            .build().unwrap();
        assert_eq!(settings, Settings {
            number_of_trials: 12345,
            trial_settings: TrialSettings::default(),
            save_frequency: 0,
            record_max: 10000,
            verbose: 0,
            seed: None,
            threads: 1,
            output_file: None,
            overwrite: false,
            silent: true,
        });
    }
}
