param([Parameter(Mandatory=$true)][ValidateSet('v013','v014','v015','v016','v017')][string]$Version,
      [ValidateSet('validation_run','e0_validation_run')][string]$Phase = 'validation_run')
$ErrorActionPreference = 'Stop'
if ($Phase -eq 'e0_validation_run' -and $Version -ne 'v017') {
    throw 'Only V017 adds a baseline archive in this helper'
}
$remote = 'D:\004SSH\OEFormer_transfer_TBPR_PRO1_20260904\remote.ps1'
$python = 'E:\Anaconda\python.exe'
$base = 'D:\004SSH\TBPS_ViewResearch_' + $Version + '_archives_20260907'
if ((Resolve-Path -LiteralPath $base).Path -ne $base) { throw 'Unexpected archive directory' }
$prefix = if ($Phase -eq 'e0_validation_run') { 'e0' } else { 'method' }
$partial = Join-Path $base ($prefix+'_completed_archive.tar.part')
$archive = Join-Path $base ($prefix+'_completed_archive.tar')
$run = Join-Path $base $Phase
if ((Test-Path -LiteralPath $archive) -or (Test-Path -LiteralPath $run)) {
    throw 'Archive already finalized or partially extracted; inspect without overwriting'
}
$download = Get-Content -LiteralPath ($partial+'.download.json') -Raw | ConvertFrom-Json
if ($download.status -ne 'complete' -or $download.bytes -ne $download.source_identity.size -or
    (Get-Item -LiteralPath $partial).Length -ne $download.bytes) { throw 'Incomplete download' }
if ((Resolve-Path -LiteralPath $partial).Path -ne $partial) { throw 'Unexpected partial path' }
$expectedRemote = '/dev/shm/TBPS_ViewResearch_' + $Version + '_20260907/'+$prefix+'_completed_archive.tar'
if ($download.source_identity.source -ne $expectedRemote) { throw 'Wrong archive source' }
$entries = & tar -tf $partial
if ($LASTEXITCODE -ne 0) { throw 'Archive listing failed' }
foreach ($entry in $entries) {
    if (-not $entry.StartsWith($Phase+'/') -or $entry.Contains('\') -or
        ($entry -split '/') -contains '..') { throw 'Unexpected archive member' }
}
if (($Phase+'/checkpoints/best.pth') -notin $entries -or
    ($Phase+'/checkpoints/last.pth') -notin $entries) { throw 'Missing checkpoints' }
Rename-Item -LiteralPath $partial -NewName ($prefix+'_completed_archive.tar')
& tar -xf $archive -C $base
if ($LASTEXITCODE -ne 0) { throw 'Extraction failed; preserve archive' }
& $python (Join-Path $PSScriptRoot 'verify_completed_archive.py') --archive-dir $run
if ($LASTEXITCODE -ne 0) { throw 'Local restoration verification failed; preserve server copy' }
$remoteRoot = '/root/autodl-tmp/TBPS_ViewResearch_' + $Version + '_20260907'
$cleanupRoot = '/root/autodl-tmp/TBPS_ViewResearch_last_cleanup_20260907'
$name = 'TBPS_ViewResearch_' + $Version + '_20260907_'+$Phase
$sourceRemote = $cleanupRoot+'/'+$name+'_source.json'
$command = @"
/root/autodl-tmp/envs/tbpsclip_official/bin/python - <<'PY'
from pathlib import Path
import json
run=Path('$remoteRoot/$Phase')
path=run/'checkpoints/last.pth'
if path.resolve()!=path or path.is_symlink(): raise ValueError('Unexpected checkpoint path')
result=json.loads((run/'result.json').read_text())
if result['status']!='complete': raise ValueError('Run incomplete')
st=path.stat()
source=dict(root=run.parent.name,path=str(path),run_dir=str(run),bytes=st.st_size,
    mtime_ns=st.st_mtime_ns,status=result['status'],experiment=result['experiment'])
with open('$sourceRemote','x') as f: json.dump(source,f,indent=2)
print(json.dumps(source))
PY
"@
& $remote exec $command
if ($LASTEXITCODE -ne 0) { throw 'Source metadata capture failed' }
& $remote get $sourceRemote (Join-Path $run 'source.json')
if ($LASTEXITCODE -ne 0) { throw 'Source metadata transfer failed' }
& $python (Join-Path $PSScriptRoot 'verify_last_before_cleanup.py') --archive-dir $run --checkpoint-subdir checkpoints
if ($LASTEXITCODE -ne 0) { throw 'Full tensor/optimizer verification failed; source preserved' }
$receiptRemote = $cleanupRoot+'/'+$name+'_verification.json'
& $remote put (Join-Path $run 'verification.json') $receiptRemote
if ($LASTEXITCODE -ne 0) { throw 'Receipt transfer failed' }
& $remote exec ('/root/autodl-tmp/envs/tbpsclip_official/bin/python '+$cleanupRoot+'/cleanup_verified_last_remote.py --verified-receipt '+$receiptRemote+' --output-dir '+$cleanupRoot+' --execute-user-approved-cleanup')
if ($LASTEXITCODE -ne 0) { throw 'Remote safety guard stopped cleanup' }
& $remote get ($cleanupRoot+'/'+$name+'.json') (Join-Path $run 'cleanup_receipt.json')
if ($LASTEXITCODE -ne 0) { throw 'Receipt retrieval failed; inspect remote record before retrying' }
Write-Output ('Verified local best/full-last archive; redundant server last released: '+$Version)
