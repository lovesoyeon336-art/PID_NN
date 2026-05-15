# matlab-run.ps1 - Send command to running MATLAB desktop via COM
# Usage: powershell -File matlab-run.ps1 "plot_results()"
param([string]$code)

if (-not $code) {
    Write-Host "Usage: matlab-run.ps1 ""<MATLAB code>"""
    exit 1
}

try {
    $matlab = [Runtime.InteropServices.Marshal]::GetActiveObject('Matlab.Application.Single')
} catch {
    try {
        $matlab = [Runtime.InteropServices.Marshal]::GetActiveObject('Matlab.Application')
    } catch {
        Write-Host "ERROR: No running MATLAB desktop found. Please open MATLAB first."
        exit 1
    }
}

$result = $matlab.Execute($code)
if ($result) { Write-Host $result }
