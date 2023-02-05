use anyhow::Context;
use serde::Serialize;
use std::io::Write;

/// Writes data in JSON format to stdout
pub fn write_json(data: &impl Serialize) -> Result<(), anyhow::Error> {
    let mut writer = std::io::stdout();
    serde_json::to_writer(&mut writer, data).context("data should be writable as JSON")?;
    writer.write_all(b"\n")?;
    Ok(())
}
