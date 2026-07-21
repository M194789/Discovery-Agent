$dir = Join-Path $PSScriptRoot 'ablation_formal\GroupC'
$files = Get-ChildItem $dir -Filter *.json
Write-Host ('Total: ' + $files.Count)
$small = $files | Where-Object { $_.Length -lt 2048 }
Write-Host ('Empty(<2KB): ' + $small.Count)
Write-Host '---latest 3---'
$latest = $files | Sort-Object LastWriteTime -Descending | Select-Object -First 3
foreach ($f in $latest) {
    Write-Host ('  ' + $f.LastWriteTime.ToString('MM-dd HH:mm:ss') + '  ' + [math]::Round($f.Length/1KB, 1) + ' KB  ' + $f.Name)
}
Write-Host '---C3 status---'
$c3 = $files | Where-Object { $_.Name -like 'C3-*' } | Sort-Object Name
foreach ($f in $c3) {
    Write-Host ('  ' + [math]::Round($f.Length/1KB, 1).ToString().PadLeft(7) + ' KB  ' + $f.Name)
}
Write-Host ''
Write-Host '===== LOG ====='
$log = Join-Path $PSScriptRoot '_targeted_runner_resume.log'
if (Test-Path $log) {
    $size = (Get-Item $log).Length
    Write-Host ('Log size: ' + [math]::Round($size/1KB, 1) + ' KB')
    Write-Host ('LastWriteTime: ' + (Get-Item $log).LastWriteTime.ToString('MM-dd HH:mm:ss'))
    Write-Host '---tail 50 lines---'
    Get-Content $log -Tail 50
} else {
    Write-Host 'Log not found'
}
Write-Host ''
Write-Host '===== PROC ====='
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Select-Object ProcessId,
        @{n='MemMB'; e={[math]::Round($_.WorkingSetSize/1MB, 0)}},
        @{n='Started'; e={$_.CreationDate.ToString('MM-dd HH:mm:ss')}} |
    Format-Table -AutoSize
Write-Host ''
Write-Host '===== PROXY ====='
$listen = Get-NetTCPConnection -LocalPort 7897 -State Listen -ErrorAction SilentlyContinue
if ($listen) { Write-Host 'Proxy port 7897: LISTENING' } else { Write-Host 'Proxy port 7897: NOT listening' }
