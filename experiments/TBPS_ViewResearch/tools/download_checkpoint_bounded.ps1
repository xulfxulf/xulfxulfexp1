param([Parameter(Mandatory=$true)][string]$Source,
      [Parameter(Mandatory=$true)][string]$Destination)
$ErrorActionPreference = 'Stop'
$raw = Get-Content -LiteralPath 'D:\004SSH\paratera_3090_login_guide.md' -Raw -Encoding UTF8
$section = [regex]::Match($raw, '(?ms)^## [^\r\n]*\bTBPR-PRO1\b[^\r\n]*\r?\n(?<body>.*?)(?=^## |\z)')
if (-not $section.Success) { throw 'Server entry missing' }
$label = [string][char]0x5bc6 + [char]0x7801
$match = [regex]::Match($section.Groups['body'].Value, '(?m)^-\s*(?:SSH\s*)?' + $label + '[' + [char]0xff1a + ':]\s*`?([^`\r\n]+)')
if (-not $match.Success) { throw 'Credential entry missing' }
$env:TBPR_PRO1_SSH_PASSWORD = $match.Groups[1].Value.Trim()
$env:PYTHONWARNINGS = 'ignore'
try {
    & 'E:\Anaconda\python.exe' (Join-Path $PSScriptRoot 'download_checkpoint_bounded.py') --source $Source --destination $Destination
    $code = $LASTEXITCODE
} finally {
    Remove-Item Env:TBPR_PRO1_SSH_PASSWORD -ErrorAction SilentlyContinue
}
exit $code
