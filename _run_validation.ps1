# Run validation and write output to file
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
Set-Location $PSScriptRoot
python -u _validate_ablation.py 2>&1 | Out-File -FilePath '_validation_report.txt' -Encoding utf8
Write-Host 'Done. Reading report:'
Get-Content '_validation_report.txt'
