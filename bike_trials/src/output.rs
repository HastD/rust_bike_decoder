use serde::Serialize;
use std::{
    fmt::Debug,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::PathBuf,
};
use thiserror::Error;
use uuid::Uuid;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum OutputTo {
    #[default]
    Stdout,
    File(PathBuf),
    Void,
}

/// Serializes data in JSON format to specified output location
pub fn write_json<D>(output_to: &OutputTo, data: &D) -> Result<(), OutputError>
where
    D: Debug + Serialize + ?Sized,
{
    let result = match Writer::new(output_to) {
        Some(Ok(mut writer)) => writer.write_json(data),
        Some(Err(e)) => Err(e),
        None => return Ok(()),
    };
    if result.is_err() {
        eprintln!("ERROR: failed to write JSON data; dumping to stderr.");
        if write_fallback(io::stderr(), data).is_err() {
            eprintln!("ERROR: fallback also failed; data may have been lost.");
        }
    }
    result
}

pub fn check_writable(output: &OutputTo, overwrite: bool) -> Result<(), OutputError> {
    if let OutputTo::File(path) = output {
        if !overwrite
            && path.try_exists().map_err(OutputError::Inaccessible)?
            && fs::metadata(path).map_err(OutputError::BadMetadata)?.len() > 0
        {
            // If file already exists and is nonempty, copy its contents to a backup file
            fs::copy(
                path,
                format!("{}-backup-{}", path.display(), Uuid::new_v4()),
            )
            .map_err(OutputError::FailedBackup)?;
        }
        File::create(path)
            .map_err(OutputError::NotWritable)?
            .write_all(b"")
            .map_err(OutputError::NotWritable)?;
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum OutputError {
    #[error("Output file path should be accessible")]
    Inaccessible(io::Error),
    #[error("Output file metadata should be readable")]
    BadMetadata(io::Error),
    #[error("Should be able to back up existing file")]
    FailedBackup(io::Error),
    #[error("Output stream should be writable")]
    NotWritable(io::Error),
    #[error("data should be writable as JSON")]
    JsonNotWritable(serde_json::Error),
}

#[derive(Debug)]
enum Writer {
    Stdout(io::Stdout),
    File(BufWriter<File>),
}

impl Writer {
    fn new(output: &OutputTo) -> Option<Result<Self, OutputError>> {
        match output {
            OutputTo::Stdout => Some(Ok(io::stdout().into())),
            OutputTo::File(filename) => Some(
                File::create(filename)
                    .map(Self::from)
                    .map_err(OutputError::NotWritable),
            ),
            OutputTo::Void => None,
        }
    }

    fn write_json<D>(&mut self, data: &D) -> Result<(), OutputError>
    where
        D: Serialize + ?Sized,
    {
        serde_json::to_writer(&mut *self, data).map_err(OutputError::JsonNotWritable)?;
        self.write_all(b"\n").map_err(OutputError::NotWritable)?;
        self.flush().map_err(OutputError::NotWritable)?;
        Ok(())
    }
}

impl Write for Writer {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::Stdout(stdout) => stdout.write(buf),
            Self::File(file) => file.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Stdout(stdout) => stdout.flush(),
            Self::File(file) => file.flush(),
        }
    }
}

impl From<io::Stdout> for Writer {
    fn from(stdout: io::Stdout) -> Self {
        Self::Stdout(stdout)
    }
}

impl From<File> for Writer {
    fn from(file: File) -> Self {
        Self::File(BufWriter::new(file))
    }
}

fn write_fallback<W, D>(mut writer: W, data: &D) -> Result<(), io::Error>
where
    W: Write,
    D: Debug + Serialize + ?Sized,
{
    if let Ok(json_str) = serde_json::to_string(data) {
        writeln!(writer, "{json_str}")?;
    } else {
        writeln!(writer, "{data:?}")?;
    }
    Ok(())
}
