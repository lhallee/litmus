# Sample the memory and processor time of every running process with a given name, one line per sample per process:
#   unix_ms,pid,working_set,peak_working_set,private_bytes,cpu_seconds
# The memory is read from the operating system's own counters for the process (nothing is inferred). It runs until it is
# stopped, so it can watch short-lived processes (one per document) as well as a long one. It names and reads no other process.
param(
  [Parameter(Mandatory = $true)][string]$Name,
  [int]$IntervalMs = 50
)
$ErrorActionPreference = 'SilentlyContinue'
$out = [Console]::Out
while ($true) {
  $now = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
  foreach ($p in Get-Process -Name $Name) {
    $cpu = $p.TotalProcessorTime.TotalSeconds.ToString('F4', [Globalization.CultureInfo]::InvariantCulture)
    $out.WriteLine("$now,$($p.Id),$($p.WorkingSet64),$($p.PeakWorkingSet64),$($p.PrivateMemorySize64),$cpu")
  }
  $out.Flush()
  Start-Sleep -Milliseconds $IntervalMs
}
