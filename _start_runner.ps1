# 启动定向消融实验 runner
# 如果你的网络需要代理才能访问 LLM API，请取消下面两行注释并填入你的代理地址
# If your network needs a proxy to reach the LLM API, uncomment and set your proxy address:
# $env:HTTPS_PROXY = 'http://127.0.0.1:7890'
# $env:HTTP_PROXY  = 'http://127.0.0.1:7890'
$env:PYTHONIOENCODING = 'utf-8'

Set-Location $PSScriptRoot

# 验证代理变量已设置
Write-Host ('HTTPS_PROXY = ' + $env:HTTPS_PROXY)
Write-Host ('HTTP_PROXY  = ' + $env:HTTP_PROXY)

# 清理旧 log（如被占用就跳过）
$log = Join-Path $PSScriptRoot '_targeted_runner_resume.log'
$err = Join-Path $PSScriptRoot '_targeted_runner_resume.err'
if (Test-Path $log) { try { Remove-Item $log -Force -ErrorAction Stop } catch { Write-Host 'log busy, will overwrite' } }
if (Test-Path $err) { try { Remove-Item $err -Force -ErrorAction Stop } catch {} }

# 启动子进程（继承当前 shell 的环境变量包括 HTTPS_PROXY）
$proc = Start-Process -FilePath 'python' `
    -ArgumentList '-u','_targeted_runner.py' `
    -RedirectStandardOutput $log `
    -RedirectStandardError  $err `
    -PassThru -WindowStyle Hidden

Write-Host ('PID = ' + $proc.Id)
Write-Host ('Started at ' + $proc.StartTime)
Start-Sleep -Seconds 4
Write-Host '---initial log tail---'
if (Test-Path $log) { Get-Content $log -Tail 20 } else { Write-Host '(log not yet written)' }
