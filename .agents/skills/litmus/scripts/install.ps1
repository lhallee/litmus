<#
.SYNOPSIS
Installs the `litmus` command line on Windows, from this project's GitHub releases and nowhere else.

.DESCRIPTION
Needs no elevation. Downloads litmus.exe and the release's published file list (SHA256SUMS), checks the program
against that list, and installs it only when the check passes; a program that does not match is never installed.
The program has the model built in, so one file is the whole install. Works in Windows PowerShell 5.1 and PowerShell 7.
Run it with: powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1  (the policy applies to that one process).

.PARAMETER Dir
Where litmus.exe goes. Default: %LOCALAPPDATA%\Litmus\bin.

.PARAMETER Version
A release tag such as v0.1.0. Default: the latest release.

.PARAMETER FromDir
Install from a folder that holds litmus.exe and SHA256SUMS instead of downloading (no network).

.PARAMETER AddToPath
Also add Dir to this user's PATH (the one registry value HKCU\Environment\Path). Without it nothing but Dir\litmus.exe
is written and the script prints how to run the program by its full path.
#>
[CmdletBinding()]
param(
    [string]$Dir = (Join-Path $env:LOCALAPPDATA 'Litmus\bin'),
    [string]$Version = 'latest',
    [string]$FromDir = '',
    [switch]$AddToPath
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Repo = 'lhallee/litmus'
$Docs = "https://github.com/$Repo/blob/main/docs/USE_WITH_AGENTS.md"
$Asset = 'litmus.exe'

function Stop-Install([string]$Message, [int]$Code = 1) {
    [Console]::Error.WriteLine("litmus install: $Message")
    exit $Code
}

# Windows ARM is not tested, and a 32-bit PowerShell on 64-bit Windows reports x86 here, so look at both variables.
$arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
if ($arch -ne 'AMD64') {
    Stop-Install "no prebuilt program for Windows on $arch. Build it from source (cargo build --release --locked in engine, with a model folder): $Docs" 2
}

if ($Version -eq 'latest') {
    $Base = "https://github.com/$Repo/releases/latest/download"
} elseif ($Version -match '^[A-Za-z0-9._-]+$') {
    $Base = "https://github.com/$Repo/releases/download/$Version"
} else {
    Stop-Install '-Version takes a release tag such as v0.1.0' 2
}

$work = Join-Path ([IO.Path]::GetTempPath()) ("litmus-install-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work | Out-Null
try {
    # Windows PowerShell 5.1 may offer only old TLS versions by default.
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    foreach ($name in @('SHA256SUMS', $Asset)) {
        $target = Join-Path $work $name
        if ($FromDir) {
            $source = Join-Path $FromDir $name
            if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { Stop-Install "$FromDir has no $name" }
            Copy-Item -LiteralPath $source -Destination $target
        } else {
            try {
                Invoke-WebRequest -UseBasicParsing -Uri "$Base/$name" -OutFile $target
            } catch {
                Stop-Install "cannot download $Base/$name (is there a release with this program? see $Docs for the build from source): $($_.Exception.Message)"
            }
        }
    }

    $expected = $null
    foreach ($line in Get-Content -LiteralPath (Join-Path $work 'SHA256SUMS')) {
        if ($line -match '^([0-9A-Fa-f]{64})\s+\*?(.+?)\s*$' -and $Matches[2] -eq $Asset) {
            $expected = $Matches[1].ToLowerInvariant()
            break
        }
    }
    if (-not $expected) { Stop-Install "the release's file list does not name $Asset; nothing was installed" }
    $actual = (Get-FileHash -LiteralPath (Join-Path $work $Asset) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { Stop-Install "the download does not match the release's published file list; nothing was installed" }

    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
    $Dir = (Resolve-Path -LiteralPath $Dir).Path
    $destination = Join-Path $Dir 'litmus.exe'
    $replaced = if (Test-Path -LiteralPath $destination) { ' (replacing the earlier copy)' } else { '' }
    $staged = Join-Path $Dir ".litmus.new.$PID.exe"
    try {
        Copy-Item -LiteralPath (Join-Path $work $Asset) -Destination $staged -Force
        Move-Item -LiteralPath $staged -Destination $destination -Force
    } catch {
        if (Test-Path -LiteralPath $staged) { Remove-Item -LiteralPath $staged -Force }
        Stop-Install "cannot write $destination (is an earlier litmus.exe still running, for example as an agent's server?): $($_.Exception.Message)"
    }

    & $destination --help *> $null
    if ($LASTEXITCODE -ne 0) {
        Stop-Install "installed $destination but it does not run on this computer; remove it and build from source ($Docs)" 3
    }

    Write-Output "Installed $destination$replaced"
    Write-Output "  from:  $(if ($FromDir) { $FromDir } else { $Base }) ($Asset), checked against the release's published file list"
    Write-Output '  model: built in, nothing else to download'
    $onPath = ($env:Path -split ';') -contains $Dir
    if ($AddToPath) {
        $user = [Environment]::GetEnvironmentVariable('Path', 'User')
        if (-not (($user -split ';') -contains $Dir)) {
            [Environment]::SetEnvironmentVariable('Path', ($(if ($user) { "$user;" } else { '' }) + $Dir), 'User')
        }
        Write-Output "  files: wrote $destination and added $Dir to this user's PATH; open a new terminal for it to take effect"
    } else {
        Write-Output "  files: wrote only $destination; no PATH or setting was changed"
        if ($onPath) {
            Write-Output "  path:  $Dir is on PATH, so litmus runs by name"
        } else {
            Write-Output "  path:  $Dir is not on PATH. Run it as `"$destination`", or install again with -AddToPath"
        }
    }
    Write-Output "Next: register the server with your agent ($Docs), for example:"
    Write-Output "  claude mcp add litmus -- `"$destination`" mcp"
} finally {
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}
