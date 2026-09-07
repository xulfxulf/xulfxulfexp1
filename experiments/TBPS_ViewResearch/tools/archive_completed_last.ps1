param(
    [string]$ArchiveRoot = 'D:\004SSH\TBPS_ViewResearch_last_archives_20260907'
)
$ErrorActionPreference = 'Stop'
$remote = 'D:\004SSH\OEFormer_transfer_TBPR_PRO1_20260904\remote.ps1'
$python = 'E:\Anaconda\python.exe'
$verifier = Join-Path $PSScriptRoot 'verify_last_before_cleanup.py'
$root = (Resolve-Path -LiteralPath $ArchiveRoot).Path
if ($root -ne 'D:\004SSH\TBPS_ViewResearch_last_archives_20260907') {
    throw 'Unexpected archive root'
}
$rows = Get-Content -LiteralPath (Join-Path $root 'inventory.json') -Raw | ConvertFrom-Json
foreach ($row in $rows) {
    $phase = ($row.run_dir -split '/')[-1]
    if ($row.root -notmatch '^TBPS_ViewResearch_v00[1-7]_20260907(_r2)?$' -or
        $phase -notin @('validation_run', 'e0_validation_run')) {
        throw 'Unexpected inventory entry'
    }
    $name = $row.root + '_' + $phase
    $destination = Join-Path $root $name
    if (Test-Path -LiteralPath $destination) { throw "Preserve existing archive: $destination" }
    New-Item -ItemType Directory -Path $destination | Out-Null
    $row | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $destination 'source.json') -Encoding utf8
    & $remote get ($row.run_dir + '/result.json') (Join-Path $destination 'result.json')
    if ($LASTEXITCODE -ne 0) { throw 'Result transfer failed; source preserved' }
    $partial = Join-Path $destination 'last.pth.part'
    Write-Output "Downloading $name ($($row.bytes) bytes)"
    & $remote get $row.path $partial
    if ($LASTEXITCODE -ne 0) { throw 'Checkpoint transfer failed; source preserved' }
    if ((Get-Item -LiteralPath $partial).Length -ne $row.bytes) { throw 'Incomplete transfer' }
    $resolved = (Resolve-Path -LiteralPath $partial).Path
    if (-not $resolved.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Partial archive outside intended local directory'
    }
    Rename-Item -LiteralPath $resolved -NewName 'last.pth'
    & $python $verifier --archive-dir $destination
    if ($LASTEXITCODE -ne 0) { throw 'Local restore verification failed; source preserved' }
    $receiptRemote = '/root/autodl-tmp/TBPS_ViewResearch_last_cleanup_20260907/' + $name + '_verification.json'
    & $remote put (Join-Path $destination 'verification.json') $receiptRemote
    if ($LASTEXITCODE -ne 0) { throw 'Verification receipt transfer failed; source preserved' }
    $command = '/root/autodl-tmp/envs/tbpsclip_official/bin/python /root/autodl-tmp/TBPS_ViewResearch_last_cleanup_20260907/cleanup_verified_last_remote.py --verified-receipt ' + $receiptRemote + ' --output-dir /root/autodl-tmp/TBPS_ViewResearch_last_cleanup_20260907 --execute-user-approved-cleanup'
    & $remote exec $command
    if ($LASTEXITCODE -ne 0) { throw 'Remote safety check stopped cleanup' }
    & $remote get ('/root/autodl-tmp/TBPS_ViewResearch_last_cleanup_20260907/' + $name + '.json') (Join-Path $destination 'cleanup_receipt.json')
    if ($LASTEXITCODE -ne 0) { throw 'Cleanup receipt download failed; inspect remote receipt before retrying' }
    Write-Output "Archived, verified and released: $name"
}
Write-Output 'All authorized completed last checkpoints archived and released.'
