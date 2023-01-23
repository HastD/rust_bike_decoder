use anyhow::{Context, Result};
use bike_decoder::{
    keys::{FilterError, Key, KeyFilter},
    ncw::NearCodewordClass,
    random::Seed,
};
use clap::Parser;
use derive_builder::Builder;
use getset::{CopyGetters, Getters};
use std::{num::NonZeroU64, path::PathBuf};
use thiserror::Error;

#[derive(Clone, Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(short = 'N', long, help = "Number of trials (required)")]
    number: f64, // parsed as scientific notation to usize
    #[arg(short, long, default_value_t=0, value_parser=clap::value_parser!(i8).range(-1..=3),
        help="Weak key filter (-1: non-weak only; 0: no filter; 1-3: type 1-3 only)")]
    weak_keys: i8,
    #[arg(short='T', long, default_value_t=3, value_parser=clap::value_parser!(u8).range(3..),
        requires="weak_keys", help="Weak key threshold")]
    weak_key_threshold: u8,
    #[arg(long, help = "Always use the specified key (in JSON format)")]
    fixed_key: Option<String>,
    #[arg(
        short = 'S',
        long,
        help = "Use error vectors from near-codeword set A_{t,l}(S)"
    )]
    ncw: Option<NearCodewordClass>,
    #[arg(
        short = 'l',
        long,
        help = "Overlap parameter l in A_{t,l}(S)",
        requires = "ncw"
    )]
    ncw_overlap: Option<usize>,
    #[arg(short, long, help = "Output file [default: stdout]")]
    output: Option<String>,
    #[arg(
        long,
        help = "If output file already exists, overwrite without creating backup"
    )]
    overwrite: bool,
    #[arg(
        long,
        help = "Run in parallel with automatically chosen number of threads"
    )]
    parallel: bool,
    #[arg(
        short,
        long,
        default_value_t = 10000.0,
        help = "Max number of decoding failures recorded"
    )]
    recordmax: f64, // parsed as scientific notation to usize
    #[arg(short, long, help = "Save to disk frequency [default: only at end]")]
    savefreq: Option<f64>, // parsed as scientific notation to usize
    #[arg(
        long,
        help = "Specify PRNG seed as 256-bit hex string [default: random]"
    )]
    seed: Option<String>,
    #[arg(long, conflicts_with_all=["parallel", "threads"],
        help="Initialize PRNG to match specified thread index (single-threaded only)")]
    seed_index: Option<u32>,
    #[arg(long, help = "Set number of threads (ignores --parallel)")]
    threads: Option<usize>,
    #[arg(short, long, action = clap::ArgAction::Count,
        help="Print statistics and/or decoding failures [repeat for more verbose, max 3]")]
    verbose: u8,
}

#[derive(Builder, Clone, CopyGetters, Debug, Eq, Getters, PartialEq)]
pub struct Settings {
    #[getset(get_copy = "pub")]
    num_trials: u64,
    #[builder(default)]
    #[getset(get = "pub")]
    trial_settings: TrialSettings,
    #[builder(default)]
    save_frequency: Option<NonZeroU64>,
    #[builder(default = "10000")]
    #[getset(get_copy = "pub")]
    record_max: usize,
    #[builder(default)]
    #[getset(get_copy = "pub")]
    verbose: u8,
    #[builder(default)]
    #[getset(get_copy = "pub")]
    seed: Option<Seed>,
    #[builder(default)]
    #[getset(get_copy = "pub")]
    seed_index: Option<u32>,
    #[builder(default = "1")]
    #[getset(get_copy = "pub")]
    threads: usize,
    #[builder(default)]
    #[getset(get = "pub")]
    output: OutputTo,
    #[builder(default)]
    #[getset(get_copy = "pub")]
    overwrite: bool,
}

impl Settings {
    const MIN_SAVE_FREQUENCY: u64 = 10000;
    const MAX_THREAD_COUNT: usize = 1024;

    pub fn from_args(args: Args) -> Result<Self> {
        let settings = Self {
            num_trials: args.number as u64,
            trial_settings: TrialSettings::new(
                KeyFilter::new(args.weak_keys, args.weak_key_threshold)?,
                args.fixed_key
                    .as_deref()
                    .map(serde_json::from_str)
                    .transpose()
                    .context("--fixed-key should be valid JSON representing a key")?
                    .map(Key::sorted),
                args.ncw,
                args.ncw_overlap,
            )?,
            save_frequency: args
                .savefreq
                .map(|s| s as u64)
                .map(|s| s.max(Self::MIN_SAVE_FREQUENCY))
                .and_then(NonZeroU64::new),
            record_max: args.recordmax as usize,
            verbose: args.verbose,
            seed: args
                .seed
                .as_deref()
                .map(Seed::try_from)
                .transpose()
                .context("--seed should be 256-bit hex string")?,
            seed_index: args.seed_index.map(|seed_idx| {
                if seed_idx >= 1 << 24 {
                    eprintln!("Warning: very large PRNG seed index will be slow to initialize.");
                }
                seed_idx
            }),
            // Default if --threads not specified:
            // * If --parallel flag not set, settings.threads = 1
            // * If --parallel flag set, settings.threads = 0, which tells
            //   Rayon to automatically determine the number of threads.
            threads: args.threads.map_or_else(
                || usize::from(!args.parallel),
                |threads| threads.clamp(1, Self::MAX_THREAD_COUNT),
            ),
            output: args
                .output
                .map_or(OutputTo::Stdout, |path| OutputTo::File(path.into())),
            overwrite: args.overwrite,
        };
        Ok(settings)
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
    pub fn save_frequency(&self) -> u64 {
        self.save_frequency.map_or(self.num_trials, u64::from)
    }

    #[inline]
    pub fn parallel(&self) -> bool {
        self.threads != 1
    }
}

#[derive(Clone, CopyGetters, Debug, Default, PartialEq, Eq)]
pub struct TrialSettings {
    #[getset(get_copy = "pub")]
    key_filter: KeyFilter,
    fixed_key: Option<Key>,
    #[getset(get_copy = "pub")]
    ncw_class: Option<NearCodewordClass>,
    #[getset(get_copy = "pub")]
    ncw_overlap: Option<usize>,
}

impl TrialSettings {
    pub fn new(
        key_filter: KeyFilter,
        fixed_key: Option<Key>,
        ncw_class: Option<NearCodewordClass>,
        ncw_overlap: Option<usize>,
    ) -> Result<Self> {
        if let Some(key) = fixed_key.as_ref() {
            key.validate()
                .context("--fixed-key must specify valid key support")?;
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
        Ok(Self {
            key_filter,
            fixed_key,
            ncw_class,
            ncw_overlap,
        })
    }

    #[inline]
    pub fn fixed_key(&self) -> Option<&Key> {
        self.fixed_key.as_ref()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum OutputTo {
    #[default]
    Stdout,
    File(PathBuf),
    Void,
}

impl OutputTo {
    #[inline]
    pub fn is_file(&self) -> bool {
        matches!(*self, Self::File(_))
    }
}

#[derive(Copy, Clone, Debug, Error)]
pub enum SettingsError {
    #[error(transparent)]
    InvalidFilter(#[from] FilterError),
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
            fixed_key: Some(
                r#"{"h0":[6,25,77,145,165,212,230,232,247,261,306,341,449,466,493],
                "h1":[35,108,119,159,160,163,221,246,249,286,310,360,484,559,580]}"#
                    .to_string(),
            ),
            ncw: Some(NearCodewordClass::C),
            ncw_overlap: Some(7),
            output: Some("test/path/to/file.json".to_string()),
            overwrite: true,
            parallel: true,
            recordmax: 123.4,
            savefreq: Some(50.0),
            seed: Some(
                "874a5940435d8a5462d8579af9f4cad2a737880dfb13620c5257a60ffaaae6cf".to_string(),
            ),
            seed_index: None,
            threads: Some(usize::MAX),
            verbose: 2,
        };
        let mut args2 = args.clone();
        args2.savefreq = None;
        let settings = Settings::from_args(args).unwrap();
        assert_eq!(settings.num_trials, 17500);
        assert_eq!(settings.trial_settings.key_filter, KeyFilter::NonWeak(4));
        assert_eq!(
            settings.trial_settings.fixed_key,
            Some(
                Key::from_support(
                    [6, 25, 77, 145, 165, 212, 230, 232, 247, 261, 306, 341, 449, 466, 493],
                    [35, 108, 119, 159, 160, 163, 221, 246, 249, 286, 310, 360, 484, 559, 580]
                )
                .unwrap()
            )
        );
        assert_eq!(
            settings.trial_settings.ncw_class,
            Some(NearCodewordClass::C)
        );
        assert_eq!(settings.trial_settings.ncw_overlap, Some(7));
        assert_eq!(settings.save_frequency(), Settings::MIN_SAVE_FREQUENCY);
        assert_eq!(settings.record_max, 123);
        assert_eq!(settings.verbose, 2);
        assert_eq!(
            settings.seed,
            Some(Seed::new([
                135, 74, 89, 64, 67, 93, 138, 84, 98, 216, 87, 154, 249, 244, 202, 210, 167, 55,
                136, 13, 251, 19, 98, 12, 82, 87, 166, 15, 250, 170, 230, 207
            ]))
        );
        assert!(settings.seed_index().is_none());
        assert_eq!(settings.threads, Settings::MAX_THREAD_COUNT);
        assert_eq!(
            settings.output,
            OutputTo::File(PathBuf::from("test/path/to/file.json"))
        );
        assert!(settings.overwrite);
        let settings2 = Settings::from_args(args2).unwrap();
        assert_eq!(settings2.save_frequency(), settings2.num_trials());
    }

    #[test]
    fn settings_builder() {
        let settings = SettingsBuilder::default()
            .num_trials(12345)
            .output(OutputTo::Void)
            .build()
            .unwrap();
        assert_eq!(
            settings,
            Settings {
                num_trials: 12345,
                trial_settings: TrialSettings::default(),
                save_frequency: None,
                record_max: 10000,
                verbose: 0,
                seed: None,
                seed_index: None,
                threads: 1,
                output: OutputTo::Void,
                overwrite: false,
            }
        );
        assert_eq!(settings.save_frequency(), settings.num_trials());
    }
}
