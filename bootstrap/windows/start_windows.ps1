param(
    [string]$LaunchUrl = "https://www.python.org/downloads/windows/"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName PresentationFramework | Out-Null

$BootstrapRoot = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent $BootstrapRoot
Set-Location $ProjectRoot
$VariantFile = Join-Path $ProjectRoot "output\.startup_variant"

function Get-StoredVariant {
    if (-not (Test-Path $VariantFile)) {
        return $null
    }

    $value = (Get-Content $VariantFile -Raw).Trim().ToLowerInvariant()
    if ($value -in @("cpu", "cuda")) {
        return $value
    }

    return $null
}

function Set-StoredVariant {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Variant
    )

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $VariantFile) | Out-Null
    Set-Content -Path $VariantFile -Value $Variant -Encoding UTF8
}

function Get-PythonCommand {
    $candidates = @(
        @{ Exe = "py"; Args = @("-3") },
        @{ Exe = "python"; Args = @() },
        @{ Exe = "python3"; Args = @() }
    )

    foreach ($candidate in $candidates) {
        if (Get-Command $candidate.Exe -ErrorAction SilentlyContinue) {
            return $candidate
        }
    }

    return $null
}

function Show-PythonMissingDialog {
    $message = @"
Python 3.10 or newer is required to start ImagePlot-CLIP.

Choose Yes to try the automatic Windows installer, or No to open the official download page.
"@

    $result = [System.Windows.MessageBox]::Show(
        $message,
        "ImagePlot-CLIP",
        [System.Windows.MessageBoxButton]::YesNoCancel,
        [System.Windows.MessageBoxImage]::Warning
    )

    switch ($result) {
        "Yes" {
            if (Get-Command winget -ErrorAction SilentlyContinue) {
                Start-Process winget -ArgumentList @(
                    "install",
                    "-e",
                    "--id",
                    "Python.Python.3.12",
                    "--source",
                    "winget"
                ) | Out-Null
            }
            else {
                Start-Process $LaunchUrl | Out-Null
            }
        }
        "No" {
            Start-Process $LaunchUrl | Out-Null
        }
    }
}

function Select-TorchVariant {
    $message = @"
Choose the Windows startup mode.

Yes = CPU, the safest first choice.
No  = CUDA, only for compatible NVIDIA systems.
Cancel = stop.
"@

    $result = [System.Windows.MessageBox]::Show(
        $message,
        "ImagePlot-CLIP",
        [System.Windows.MessageBoxButton]::YesNoCancel,
        [System.Windows.MessageBoxImage]::Question
    )

    switch ($result) {
        "Yes" { return "cpu" }
        "No" { return "cuda" }
        default { return $null }
    }
}

$python = Get-PythonCommand
if (-not $python) {
    Show-PythonMissingDialog
    exit 1
}

$variant = Get-StoredVariant
if (-not $variant) {
    $variant = Select-TorchVariant
    if ($variant) {
        Set-StoredVariant -Variant $variant
    }
}
if (-not $variant) {
    exit 1
}

$command = @()
if ($python.Args) {
    $command += $python.Args
}
$command += @("launcher.py", "--torch", $variant)

& $python.Exe @command
exit $LASTEXITCODE
